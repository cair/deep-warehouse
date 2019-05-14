import time
from collections import OrderedDict
import gym
from absl import flags
import tensorflow as tf
import datetime
import os
import numpy as np

from experiments.experiment_5.per_rl import utils
from experiments.experiment_5.per_rl.storage.batch_handler import DynamicBatch
from experiments.experiment_5.per_rl.utils.metrics import Metrics

FLAGS = flags.FLAGS


class Agent:
    SUPPORTED_BATCH_MODES = ["episodic", "steps"]
    DEFAULTS = dict()
    arguments = utils.arguments

    def __init__(self,
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 policy,
                 batch_shuffle=False,
                 batch_mode: str = "episodic",
                 mini_batches: int = 1,
                 batch_size: int = 32,
                 epochs: int = 1,
                 grad_clipping=None,
                 dtype=tf.float32,
                 tensorboard_enabled=False,
                 tensorboard_path="./tb/",
                 name_prefix=None):

        hyper_parameters = utils.get_defaults(self, Agent.arguments())

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
        self.batch_mode = batch_mode
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle
        self.mini_batches = mini_batches
        self.dtype = dtype
        self.grad_clipping = grad_clipping
        self.epochs = epochs
        self.policy = policy(self)  # Initialize policy

        self._tensorboard_enabled = tensorboard_enabled
        self._tensorboard_path = tensorboard_path
        self._name_prefix = name_prefix

        self.metrics = Metrics(self)
        self.data = dict()  # Keeps track of all data per iteration. Resets after train()
        self.losses = dict()
        self.operations = OrderedDict()

        self.metrics.text("hyperparameters", tf.convert_to_tensor(utils.hyperparameters_to_table(hyper_parameters)))

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

        self.obs = None  # Last seen observation

    def add_operation(self, name, fn):
        self.operations[name] = fn

    def remove_calculation(self, name):
        del self.operations[name]

    def add_loss(self, name, lambda_fn, tb_text=None):
        self.losses[name] = lambda_fn

        if tb_text:
            """Add text on tensorboard"""
            self.metrics.text(name, tb_text)

    def remove_loss(self, name):
        del self.losses[name]

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
        self.data.update(pred)

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
        self.metrics.add("reward", kwargs["reward"], ["sum", "sum_mean"], None, episode=True, epoch=False, total=True)

        if kwargs["terminal"]:
            self.metrics.done(episode=True)
            self.metrics.summarize(["reward", "steps"])

        ready = self.batch.add(
            **self.data
        )

        if ready:  # or not self.inference_only:
            train_start = time.perf_counter()
            losses = self.train()

            """Update metrics for training"""
            self.metrics.add("total", np.mean(losses), ["mean"], "loss", epoch=True, total=True)
            self.metrics.add("backprop", time.perf_counter() - train_start, ["mean"], "time", epoch=True)

    def _backprop(self, **kwargs):
        total_loss = 0
        losses = []
        for policy_name, policy in self.policy.trainers:

            """Run all loss functions"""
            with tf.GradientTape() as tape:

                pred = policy(**kwargs)
                kwargs.update(pred)

                for loss_name, loss_fn in self.losses.items():
                    loss = loss_fn(**kwargs)

                    """Add metric for loss"""
                    self.metrics.add(policy_name + "/" + loss_name, loss, ["mean"], "loss", epoch=True, total=True)

                    """Add to total loss"""
                    total_loss += loss
                    losses.append(loss)

            """Calculate gradients"""
            grads = tape.gradient(total_loss, policy.trainable_variables)

            """Gradient Clipping"""
            if self.grad_clipping is not None:
                grads, _grad_norm = tf.clip_by_global_norm(grads, self.grad_clipping)

            """Diagnostics"""
            self.metrics.add("variance", np.mean([np.var(grad) for grad in grads]), ["mean"], "gradients", epoch=True)
            self.metrics.add("l2", np.mean([np.sqrt(np.mean(np.square(grad))) for grad in grads]), ["mean"],
                             "gradients", epoch=True)

            """Backprop"""
            policy.optimize(grads)

            """Record learning rate"""
            for optimizer_name, optimizer in policy.optimizer.items():
                self.metrics.add(policy_name + "/" + optimizer_name + "/learning-rate", optimizer.lr.numpy(), ["mean"], "hyper-parameter", epoch=True) # todo name
                optimizer.lr = optimizer.lr - (optimizer.lr * optimizer.decay)

        self.policy.optimize(None)
        return np.asarray(losses)

    def train(self, **kwargs):

        kwargs["policy"] = self.policy

        # Retrieve batch of data
        batch = self.batch.get()

        # Perform calculations prior to training
        for opname, operation in self.operations.items():
            return_op = operation(**batch, **kwargs)

            if isinstance(return_op, dict):
                for k, v in return_op.items():
                    batch[k] = v
            else:
                batch[opname] = return_op

        batch_indices = np.arange(self.batch.counter)  # We use counter because episodic wil vary in bsize.
        # Calculate mini-batch size
        for epoch in range(self.epochs):


            # Shuffle the batch indices
            if self.batch_shuffle:
                np.random.shuffle(batch_indices)

            for i in range(0, self.batch.counter, self.batch.mbsize):
                # Sample indices for the mini-batch
                mb_indexes = batch_indices[i:i + self.batch.mbsize]

                # Cast all elements to numpy arrays
                mb = {k: np.asarray(v)[mb_indexes] for k, v in batch.items()}

                losses = self._backprop(**mb, **kwargs)

        self.metrics.add("epochs", 1, ["sum"], "time/training", total=True, episode=True)
        self.metrics.done(epoch=True)

        self.batch.done()
        return losses
