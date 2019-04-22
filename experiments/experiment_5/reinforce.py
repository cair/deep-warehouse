import os


from experiments.experiment_5.agent import Agent
from experiments.experiment_5.network import PGPolicy

import tensorflow as tf
import gym
import numpy as np
from absl import flags, logging
import time

FLAGS = flags
logging.set_verbosity(logging.DEBUG)


class REINFORCE(Agent):

    DEFAULTS = dict(
        batch_mode="episodic",
        batch_size=32,
        policies=dict(
            target=lambda agent: PGPolicy(
                agent=agent,
                inference=True,
                training=True,
                optimizer=tf.keras.optimizers.Adam(lr=0.001)
            )
        )
    )

    def __init__(self,
                 gamma=0.99,
                 entropy_coef=0.01,
                 baseline=None,
                 **kwargs):

        super(REINFORCE, self).__init__(**Agent.arguments())

        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.baseline = baseline

        self.add_calculation("discounted_reward", self.discounted_returns)
        self.add_loss("policy_loss", self.policy_loss, "**Policy loss (REINFORCE)**  \nThes policy loss will oscillate with training.")

        #if entropy_coef != 0:
        #    self.add_loss("entropy_loss", self.entropy_loss)

    def predict(self, inputs):
        start = time.perf_counter()
        prediction = super().predict(inputs)

        self.metrics.add("inference_time", time.perf_counter() - start)
        return np.argmax(prediction["actions"])

    def observe(self, **kwargs):
        ready = super().observe(**kwargs)

        if ready:
            s = time.perf_counter()

            data = self.batch.flush()
            for mb in tf.data.Dataset.from_tensor_slices(data):
                self.train(**mb)

            #self.metrics.add("backprop_time", time.perf_counter() - s, "EpisodicMean")

    """
    # G is commonly refered to as the cumulative discounted rewards
    """

    def discounted_returns(self, rewards, terminals, **kwargs):
        # TODO - Checkout https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py and compare performance
        print(kwargs)

        discounted_rewards = np.zeros_like(rewards)

        cum_r = 0
        l = len(rewards)
        for i in reversed(range(0, l)):
            cum_r = rewards[i] + (cum_r * self.gamma * (1 - terminals[i]))

            discounted_rewards[i] = cum_r

        discounted_rewards = (discounted_rewards - discounted_rewards.std()) / discounted_rewards.mean()
        np.nan_to_num(discounted_rewards, copy=False)

        """Baseline implementations."""
        if self.baseline == "reward_mean":
            baseline = tf.reduce_mean(rewards)
        else:
            baseline = 0

        return discounted_rewards - baseline

    def generalized_advantage_estimation(self, values, next_value, rewards, terminals, gamma=0.99, tau=0.95):
        # TODO not really working ...
        nvalues = np.concatenate((values.numpy(), [next_value]))
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):

            delta = rewards[step] + gamma * nvalues[step + 1] * terminals[step] - nvalues[step]
            gae = delta + gamma * tau * terminals[step] * gae
            returns.insert(0, gae + nvalues[step])


        return returns

    def G(self, data, **kwargs):
        data["G"] = tf.convert_to_tensor(
            self.discounted_returns(kwargs["rewards"], kwargs["terminals"])
        )

    """
    actions: one hot vectors of actions
    G: discounted rewards (advantages)
    predicted_logits: The network's predicted logits.
    """
    def policy_loss(self, logits=None, actions=None, G=None, **kwargs):

        neg_log_p = -tf.math.log_softmax(logits) * actions
        loss = neg_log_p * G

        return tf.reduce_mean(loss)

    def entropy_loss(self, logits=None, obs=None, **kwargs):
        #entropy_loss = - tf.reduce_sum(policy_logits.logits * tf.math.log(policy_logits.logits))

        entropy_loss = -tf.reduce_sum(logits * tf.math.log(logits))
        return -entropy_loss * self.entropy_coef


