import gym
import pycallgraph
from absl import flags
import tensorflow as tf
import datetime
import os

from pycallgraph.output import GraphvizOutput

from experiments.experiment_5 import utils
from experiments.experiment_5.batch_handler import VectorBatchHandler, DynamicBatch
from experiments.experiment_5.metrics import Metrics

FLAGS = flags.FLAGS


class Agent:
    SUPPORTED_BATCH_MODES = ["episodic", "steps"]
    DEFAULTS = dict()
    arguments = utils.arguments

    def __init__(self,
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 policies: dict,
                 policy_update=dict(
                     interval=5,  # Update every 5 training epochs,
                     strategy="copy",  # "copy, mean"
                 ),
                 batch_mode: str = "episodic",
                 mini_batches: int = 1,
                 batch_size: int = 32,
                 max_grad_norm=None,
                 dtype=tf.float32,
                 tensorboard_enabled=False,
                 tensorboard_path="./tb/",
                 name_prefix=None,
                 metrics_enabled=dict(),
                 metrics_trigger="terminal",
                 inference_only=False):
        args = utils.get_defaults(self, Agent.arguments())

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
        self.policies = policies
        self.batch_mode = batch_mode
        self.batch_size = batch_size
        self.mini_batches = mini_batches
        self.dtype = dtype
        self.max_grad_norm = max_grad_norm
        self.inference_only = inference_only

        self._tensorboard_enabled = tensorboard_enabled
        self._tensorboard_path = tensorboard_path
        self._name_prefix = name_prefix

        self.metrics = Metrics(self)
        self.metrics_trigger = metrics_trigger
        self.metrics_enabled = metrics_enabled

        self.data = dict()  # Keeps track of all data per iteration. Resets after train()

        self.losses = dict()
        self.operations = dict()

        self.metrics.text("hyperparameters", tf.convert_to_tensor(utils.hyperparameters_to_table(self._hyperparameters)))

        self.policies = {
            k: v(self) for k, v in policies.items()
        }

        if batch_mode not in Agent.SUPPORTED_BATCH_MODES:
            raise NotImplementedError("The batch mode %s is not supported. Use one of the following: %s" %
                                      (batch_mode, Agent.SUPPORTED_BATCH_MODES))
        self.batch_mode = batch_mode
        """Find all policies with inference flag set. Ensure that its only 1 and assign as the inference 
        policy. """
        self.inference_policy = [x for k, x in self.policies.items() if x.inference]
        if len(self.inference_policy) != 1:
            raise ValueError("There can only be 1 policy with the flag training=False.")
        self.inference_policy = self.inference_policy[0]

        """This list contains names of policies that should be trained"""
        self.training_policies = [x for k, x in self.policies.items() if x.training]

        # Policy update. This is the strategy used when using multiple policies (ie one trainer and one predictor)
        # Settings here determine how updates should be performed.
        self.policy_update = policy_update
        self.policy_update_counter = 0
        self.policy_update_frequency = self.policy_update["interval"]
        self.policy_update_enabled = len(self.policies) > 1

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

    # @tf.function
    def predict(self, inputs):
        """
        :param inputs: DATA INPUT
        :param policy: Which policy to use. When None, self.inference_policy will be used.
        :return:
        """
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        self.data["inputs"] = inputs

        return self.inference_policy(inputs)

    def observe(self, **kwargs):
        """
        Observe the resulting transition
        :param obs1: The next state s_t+1
        :param reward: Reward yielded back from R_t = S_t(a_t)
        :param terminal: T_t = S_t(a_t) (Not really RL thing, but we keep track of Terminal states
        modify reward in some cases.
        :return:
        """
        self.metrics.add("steps", 1)
        self.data.update(**kwargs)

        if kwargs[self.metrics_trigger]:
            self.metrics.summarize()

        return self.batch.add(
            **self.data
        )

    def train(self, **kwargs):
        if self.inference_only:
            return 0

        total_loss = 0
        self.policy_update_counter += 1

        """Policy training procedure"""
        for policy in self.training_policies:

            with tf.GradientTape() as tape:

                """Pack data container"""
                kwargs["policy"] = policy

                pred = policy(**kwargs)

                """Run all calculations"""
                for opname, operation in self.operations.items():
                    kwargs[opname] = operation(**kwargs)

                """Run all loss functions"""
                for loss_name, loss_fn in self.losses.items():
                    loss = loss_fn(pred=pred, **kwargs)

                    """Add metric for loss"""
                    self.metrics.add(loss_name, loss, type="EpisodicMean")

                    """Add to total loss"""
                    total_loss += loss

            """Calculate gradients"""
            grads = tape.gradient(total_loss, policy.trainable_variables)

            if self.max_grad_norm is not None:
                grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

            """Backprop"""
            policy.optimizer.apply_gradients(zip(grads, policy.trainable_variables))

            """Record learning rate"""

            self.metrics.add("learning_rate", policy.optimizer.lr.numpy(), "EpisodicMean")

        """Policy update strategy (If applicable)."""
        if self.policy_update_enabled and self.policy_update_counter % self.policy_update_frequency == 0:
            strategy = self.policy_update["strategy"]

            if strategy == "mean":
                raise NotImplementedError("Not implemented yet")
            elif strategy == "copy":
                for policy in self.training_policies:
                    self.inference_policy.set_weights(policy.get_weights())
                self.policy_update_counter = 0
            else:
                raise NotImplementedError("The policy update strategy %s is not implemented for the BaseAgent." % strategy)

        """Update metrics for training"""
        self.metrics.add("iterations_per_episode", 1, "Sum")
        self.metrics.add("total_epochs", 1, "InfiniteSum")
        self.metrics.add("total_loss", total_loss, "Mean")
        return total_loss
