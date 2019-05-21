import time
from collections import OrderedDict
import gym
from absl import flags
import tensorflow as tf
import datetime
import os
import numpy as np

from experiments.experiment_5.per_rl import utils
from experiments.experiment_5.per_rl.agents.configuration.models import PolicyManager
from experiments.experiment_5.per_rl.storage.batch_handler import DynamicBatch
from experiments.experiment_5.per_rl.utils.metrics import Metrics

FLAGS = flags.FLAGS


def DecoratedAgent(cls, **kwargs):
    __init__ = cls.__init__

    def wrapper(self, **kwargs):
        # Copy default configuration. Update with arguments
        defaults = cls.DEFAULTS.copy() # .update(kwargs)
        defaults.update(kwargs)

        if not hasattr(self, "args"):
            setattr(self, "args", {})

        # Extract required parameters for this agent class
        args = dict()
        for param in cls.PARAMETERS:

            try:
                args[param] = defaults[param]
                #del defaults[param]  # todo?
            except KeyError:
                raise ValueError("The parameter '%s' is required for '%s' but was not found in the configuration." % (param, cls.__name__))

        args.update(getattr(self, "args"))
        setattr(self, "args", args)

        __init__(self, **defaults)

    cls.__init__ = wrapper
    return cls


class Agent:
    SUPPORTED_BATCH_MODES = ["episodic", "steps"]
    SUPPORTED_PROCESSORS = ["batch", "mini-batch", "loss", "post"]
    DEFAULTS = dict()

    def __init__(self,
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 policy,
                 policy_update,
                 batch_shuffle=False,
                 batch_mode: str = "episodic",
                 mini_batches: int = 1,
                 batch_size: int = 32,
                 epochs: int = 1,
                 grad_clipping=None,
                 dtype=tf.float32,
                 tensorboard_enabled=False,
                 tensorboard_path="./tb/",
                 name_prefix=None,
                 **kwargs):
        self.name = self.__class__.__name__

        if tensorboard_enabled:
            logdir = os.path.join(
                tensorboard_path, "%s_%s" %
                                  (
                                      self.name + name_prefix if name_prefix else self.name,
                                      datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
            )
            writer = tf.summary.create_file_writer(logdir)
            writer.set_as_default()

        """Define properties."""
        self.obs_space = obs_space
        self.action_space = action_space

        # TODO - Define as mixin?
        self.batch_mode = batch_mode
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle
        self.mini_batches = mini_batches

        self.dtype = dtype
        self.grad_clipping = grad_clipping
        self.epochs = epochs
        self.policy = PolicyManager(policy(self), **policy_update)  # Initialize policy and pass to the policy manager.

        # TODO - Define as mixin?
        self._tensorboard_enabled = tensorboard_enabled
        self._tensorboard_path = tensorboard_path
        self._name_prefix = name_prefix

        # TODO - Define as mixin?
        self.metrics = Metrics(self)
        self.data = dict()  # Keeps track of all data per iteration. Resets after train()
        self.udata = dict()  # Data storage for data processed during training. should be cleared after traini

        self.processors = {
            t: OrderedDict() for t in Agent.SUPPORTED_PROCESSORS
        }

        self.metrics.text("hyperparameters", tf.convert_to_tensor(utils.hyperparameters_to_table(self.args)))

        if batch_mode not in Agent.SUPPORTED_BATCH_MODES:
            raise NotImplementedError("The batch mode %s is not supported. Use one of the following: %s" %
                                      (batch_mode, Agent.SUPPORTED_BATCH_MODES))
        self.batch_mode = batch_mode

        self.batch = DynamicBatch(
            agent=self,
            obs_space=obs_space,
            action_space=action_space,
            batch_size=batch_size
        )

        self.epoch = 0

        self._env = None
        self._last_observation = {}


        self._train_ready = False

    def set_env(self, env):
        if self._env is not None:
            raise UserWarning("You are now overriding existing env %s. Beware!" % env.__class__.__name__)
        self._env = env

    def step(self, action):
        s1, r, t = self._env.step(action)
        self._last_observation["last_obs"] = s1

        return s1, r, t

    def _validate_processor(self, t):
        if t not in Agent.SUPPORTED_PROCESSORS:
            raise NotImplementedError("A processor with type %s does not exist (%s)" % (t, Agent.SUPPORTED_PROCESSORS))

    def clear_processors(self, t):
        self._validate_processor(t)
        self.processors[t].clear()

    def add_processor(self, name, fn, t, text=None):
        self._validate_processor(t)
        processors = self.processors[t]

        if name in processors:
            UserWarning("The processor of type %s with name %s was overridden." % (t, name))
            del processors[name]

        if text:
            """Add text on tensorboard"""
            self.metrics.text(name, text)

        processors[name] = fn

    def remove_processor(self, name, t):
        self._validate_processor(t)
        del self.processors[t][name]

    def _predict(self, inputs):
        """
        :param inputs: DATA INPUT
        :param policy: Which policy to use. When None, self.inference_policy will be used.
        :return:
        """
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        self.data["inputs"] = inputs

        pred = self.policy(inputs)
        self.data.update({
            "old_%s" % k: v for k, v in pred.items()
        })

        return pred

    def predict(self, inputs):
        start = time.perf_counter()
        pred = self._predict(inputs)
        self.metrics.add("inference", time.perf_counter() - start, ["mean"], "time", episode=True)
        return pred

    def observe(self, **kwargs):
        """
        Observe the resulting transition
        :param obs1: The next state s_t+1
        :param reward: Reward yielded back from R_t = S_t(a_t)
        :param terminal: T_t = S_t(a_t) (Not really RL thing, but we keep track of Terminal states
        modify reward in some cases.
        :return:
        """
        self.data.update(**kwargs)

        """Metrics update."""
        self.metrics.add("steps", 1, ["sum", "sum_mean"], None, episode=True, epoch=True, total=True)
        self.metrics.add("reward", kwargs["rewards"], ["sum", "sum_mean"], None, episode=True, epoch=False, total=True)

        if kwargs["terminals"]:
            self.metrics.done(episode=True)
            self.metrics.summarize(["reward", "steps"])

        self._train_ready = self.batch.add(
            **self.data
        )

    def _backprop(self, **kwargs):

        total_loss = 0
        losses = []
        for policy_train in self.policy.slaves:

            """Run all loss functions"""
            with tf.GradientTape() as tape:

                # Do prediction using current slave policy, add this to the kwargs term.
                kwargs.update(policy_train(**kwargs))

                # Do preprossessing of mini-batch data
                self._preprocessing(kwargs, ptype="mini-batch")

                # Run all loss functions
                for loss_name, loss_fn in self.processors["loss"].items():

                    # Perform loss calculation
                    loss = loss_fn(**kwargs)

                    # Add metric for current loss
                    self.metrics.add(policy_train.alias + "/" + loss_name, loss, ["mean"], "loss", epoch=True, total=True)

                    # Accumulate losses
                    total_loss += loss
                    losses.append(loss)

            # Calculate the gradient of this backward pass.
            grads = tape.gradient(total_loss, policy_train.trainable_variables)

            # Clip gradients if enabled.
            if self.grad_clipping is not None:
                grads, _grad_norm = tf.clip_by_global_norm(grads, self.grad_clipping)

            # Save the calculated gradients inside the policy instance
            policy_train.set_grads(grads)

            """Diagnostics"""
            #self.metrics.add("variance", np.mean([np.var(grad) for grad in grads]), ["mean"], "gradients", epoch=True)
            #self.metrics.add("l2", np.mean([np.sqrt(np.mean(np.square(grad))) for grad in grads]), ["mean"],
            #                 "gradients", epoch=True)

        # Optimize all of the policies
        self.policy.optimize()

        # postprocess the data
        self._preprocessing(kwargs, ptype="post")

        return np.asarray(losses)

    def _preprocessing(self, batch, ptype, **kwargs):
        preprocess_fns = self.processors[ptype]

        # Preprocessing of data prior to training
        for preprocess_name, preprocess_fn in preprocess_fns.items():
            preprocess_res = preprocess_fn(**batch, **kwargs)

            if isinstance(preprocess_res, dict):
                batch.update(preprocess_res)
            else:
                batch[preprocess_name] = preprocess_res

    #@Benchmark
    def train(self, **kwargs):
        """
        :param kwargs: Kwargs is used throughout the training process.
        During preprocessing the kwargs is filled with new data, typically advantages... etc
        :return:
        """
        if not self._train_ready:
            return False

        # Save policy into the kwargs dictionary for use in preprocessing etc.
        kwargs["policy"] = self.policy

        # Retrieve batch
        batch = self.batch.get()

        # Preprocess the data
        self._preprocessing(batch, ptype="batch", **kwargs)

        # Retrieve batch indices for this batch
        batch_indices = np.arange(self.batch.counter)

        # Run number of epochs on the batch
        for epoch in range(self.epochs):

            #  Shuffle the batch indices if set
            if self.batch_shuffle:
                np.random.shuffle(batch_indices)

            # Iterate over mini-batches
            for i in range(0, self.batch.counter, self.batch.mb_size):

                # Sample indices for the mini-batch
                mb_indexes = batch_indices[i:i + self.batch.mb_size]

                # Cast all elements to numpy arrays
                mb = {k: np.asarray(v)[mb_indexes] for k, v in batch.items()}

                losses = self._backprop(**mb, **kwargs)

        self.metrics.add("epochs", 1, ["sum"], "time/training", total=True, episode=True)
        self.metrics.done(epoch=True)

        self.batch.done()
        self.udata.clear()
        self.epoch += 1


        """Update metrics for training"""
        self.metrics.add("total", np.mean(losses), ["mean"], "loss", epoch=True, total=True)
        #self.metrics.add("backprop", time.perf_counter() - train_start, ["mean"], "time", epoch=True)

        return losses
