import gym
import pycallgraph
from absl import flags
import tensorflow as tf
import datetime
import os

from pycallgraph.output import GraphvizOutput

from experiments.experiment_5 import utils
from experiments.experiment_5.batch_handler import VectorBatchHandler
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
        self.last_predict = None
        self.loss_fns = dict()
        self.calculation = dict()
        self.calculation_list = []

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

        self.batch = VectorBatchHandler(
            agent=self,
            obs_space=obs_space,
            action_space=action_space,
            batch_size=batch_size
        )

        """Default agent calculations."""
        self.add_calculation("predict", self.calculation_predict)

    def add_calculation(self, name, fn):
        self.calculation[name] = len(self.calculation_list)
        self.calculation_list.append(fn)

    def remove_calculation(self, name):
        self.calculation_list.remove(self.calculation[name])
        del self.calculation[name]

    def add_loss(self, name, lambda_fn, tb_text=None):
        self.loss_fns[name] = lambda_fn

        if tb_text:
            """Add text on tensorboard"""
            self.metrics.text(name, tb_text)

    def remove_loss(self, name):
        del self.loss_fns[name]

    # @tf.function
    def predict(self, inputs, policy=None):
        """
        :param inputs: DATA INPUT
        :param policy: Which policy to use. When None, self.inference_policy will be used.
        :return:
        """
        inputs = tf.cast(inputs, dtype=self.dtype)

        try:
            self.last_predict = self.inference_policy(inputs) if policy is None else self.policies[policy](inputs)
        except KeyError:
            raise ValueError("There is no policy with the name %s" % policy)

        return self.last_predict

    def get_action(self, inputs):
        raise NotImplementedError("get_action is not implemented in Agent base class.")

    def observe(self, obs1, reward, terminal):
        """
        Observe the resulting transition
        :param obs1: The next state s_t+1
        :param reward: Reward yielded back from R_t = S_t(a_t)
        :param terminal: T_t = S_t(a_t) (Not really RL thing, but we keep track of Terminal states
        modify reward in some cases.
        :return:
        """
        self.metrics.add("steps", 1)
        self.metrics.add("reward", reward)

        if terminal:
            r = self.metrics.get("reward").result(),
            self.metrics.add("reward_avg", r,  type="Mean")
            self.metrics.add("reward_avg_full", r, type="InfiniteMean")
            self.metrics.summarize()

        return self.batch.add(
            obs1=tf.cast(obs1, dtype=self.dtype),
            reward=tf.cast(reward, dtype=self.dtype),
            terminal=tf.cast(terminal, tf.bool),
            increment=True
        )

    def train(self, obs, obs1, action, action_logits, rewards, terminals):
        if self.inference_only:
            return 0

        total_loss = 0
        self.policy_update_counter += 1

        """Policy training procedure"""
        for policy in self.training_policies:

            with tf.GradientTape() as tape:

                """Pack data container"""
                data = dict(
                    obs=obs,
                    obs1=obs1,
                    actions=action,
                    action_logits=action_logits,
                    rewards=rewards,
                    terminals=terminals,
                    policy=policy
                )

                """Run all calculations"""
                for calculation in self.calculation_list:
                    calculation(data=data, **data)

                """Run all loss functions"""
                for loss_name, loss_fn in self.loss_fns.items():
                    loss = loss_fn(**data)
                    loss = tf.reduce_mean(loss)

                    """Add metric for loss"""
                    self.metrics.add(loss_name, loss, type="Mean")

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
        self.metrics.add("total_loss", total_loss, "Mean")
        return total_loss

    def calculation_predict(self, data, policy=None, obs=None, **kwargs):
        data.update(policy(obs))