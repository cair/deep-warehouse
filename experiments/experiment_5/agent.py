import gym
import tensorflow as tf
import datetime
import os

from experiments.experiment_5.batch_handler import BatchHandler
from experiments.experiment_5.metrics import Metrics


class Agent:

    def __init__(self,
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 batch_size: int,
                 dtype,
                 policies,
                 tensorboard_enabled,
                 tensorboard_path,
                 name_prefix=""
                 ):
        self.last_predict = None
        self.dtype = dtype
        self.loss_fns = {}
        self.action_space = action_space
        self.metrics = Metrics(self)
        self.policies = policies

        """Initialize Policies. This is a "hack" to support pickling of models."""
        for key, policy in self.policies.items():
            self.policies[key]["model"] = self.policies[key]["model"](**self.policies[key]["args"])

        """Find all policies with inference flag set. Ensure that its only 1 and assign as the inference 
        policy. """
        self.inference_policy = [x for k, x in self.policies.items() if x["inference"]]
        if len(self.inference_policy) != 1:
            raise ValueError("There can only be 1 policy with the flag training=False.")
        self.inference_policy = self.inference_policy[0]

        """This list contains names of policies that should be trained"""
        self.training_policies = [x for k, x in self.policies.items() if x["training"]]

        self.name = self.__class__.__name__
        self.batch = BatchHandler(
            agent=self,
            obs_space=obs_space,
            action_space=action_space,
            batch_size=batch_size,
            dtype=dtype
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
            self.last_predict = self.policies[policy]["model"](inputs)
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

        for policy_spec in self.training_policies:
            policy = policy_spec["model"]

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
