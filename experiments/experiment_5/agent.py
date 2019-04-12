import gym
import tensorflow as tf
import datetime
import os
import inspect

from experiments.experiment_5 import utils
from experiments.experiment_5.batch_handler import BatchHandler
from experiments.experiment_5.metrics import Metrics


class Agent:
    SUPPORTED_BATCH_MODES = ["episodic", "steps"]
    DEFAULTS = dict()
    arguments = utils.arguments

    def __init__(self,
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 policies: dict,
                 batch_mode: str="episodic",
                 batch_size: int=32,
                 dtype=tf.float32,
                 tensorboard_enabled=False,
                 tensorboard_path="./tb/",
                 name_prefix=""):
        args = utils.get_defaults(self, Agent.arguments())

        """Define properties."""
        self.obs_space = obs_space
        self.action_space = action_space
        self.policies = policies
        self.batch_mode = batch_mode
        self.batch_size = batch_size
        self.dtype = dtype

        self._tensorboard_enabled = tensorboard_enabled
        self._tensorboard_path = tensorboard_path
        self._name_prefix = name_prefix

        self.metrics = Metrics(self)
        self.last_predict = None
        self.loss_fns = dict()

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

        self.name = self.__class__.__name__
        self.batch = BatchHandler(
            agent=self,
            obs_space=obs_space,
            action_space=action_space,
            batch_size=batch_size
        )

        if tensorboard_enabled:
            logdir = os.path.join(
                tensorboard_path, "%s-%s_%s" %
                                  (
                                      self.name,
                                      name_prefix,
                                      datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
            )
            writer = tf.summary.create_file_writer(logdir)
            writer.set_as_default()

    def add_loss(self, name, lambda_fn):
        self.loss_fns[name] = lambda_fn

    # @tf.function
    def predict(self, inputs, policy="target"):
        inputs = tf.cast(inputs, dtype=self.dtype)

        try:
            self.last_predict = self.policies[policy](inputs)
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
            self.metrics.add("reward_avg", self.metrics.get("reward").result(), type="Mean")
            self.metrics.summarize()

        return self.batch.add(
            obs1=tf.cast(obs1, dtype=self.dtype),
            reward=tf.cast(reward, dtype=self.dtype),
            terminal=tf.cast(terminal, tf.bool),
            increment=True
        )

    def train(self, observations):
        total_loss = 0

        for policy in self.training_policies:

            with tf.GradientTape() as tape:
                predicted_logits = policy(observations)

                for loss_name, loss_fn in self.loss_fns.items():
                    loss = loss_fn(predicted_logits)
                    self.metrics.add(loss_name, loss, type="Mean")
                    total_loss += loss

            grads = tape.gradient(total_loss, policy.trainable_variables)
            policy.optimizer.apply_gradients(zip(grads, policy.trainable_variables))

        self.metrics.add("iterations_per_episode", 1, "Sum")
        self.metrics.add("total_loss", total_loss, "Mean")

        return total_loss
