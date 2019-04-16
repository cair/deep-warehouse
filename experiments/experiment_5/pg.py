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
                 baseline=None,
                 **kwargs):
        super(REINFORCE, self).__init__(**Agent.arguments())

        self.gamma = gamma
        self.baseline = baseline

        self.add_loss("policy_loss",
                      lambda y, data: self.policy_loss(
                          data["actions"],
                          self.G(data["rewards"], data["terminals"]),
                          y["policy_logits"]
                      ))

    def reset(self):
        self.batch.counter = 0

    def get_action(self, observation):
        prediction = super().predict(observation)
        policy_logits = prediction["policy_logits"]

        action_sample = np.random.choice(np.arange(self.action_space), p=tf.squeeze(policy_logits).numpy())

        self.batch.add(obs=observation, action_logits=policy_logits, action=action_sample)

        return action_sample

    def observe(self, obs1, reward, terminal):
        super().observe(obs1, reward, terminal)

        if self.batch.counter == 0:
            s = time.time()

            for obs, obs1, action, action_logits, rewards, terminals in zip(*self.batch.get()):
                loss = self.train(obs, obs1, action, action_logits, rewards, terminals)

            self.metrics.add("backprop_time", time.time() - s)
            self.batch.clear()  # TODO?

    """
    # G is commonly refered to as the cumulative discounted rewards
    """

    def G(self, rewards, terminals):
        # TODO - Checkout https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py and compare performance
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
            baseline = rewards.mean()
        else:
            baseline = 0

        return discounted_rewards - baseline

    """
    A: one hot vectors of actions
    G: discounted rewards
    predicted_logits: The network's predicted logits.
    """
    def policy_loss(self, A, G, predicted_logits):

        log_action_prob = tf.keras.backend.log(
            tf.keras.backend.sum(predicted_logits * A, axis=1)
        )

        loss = - log_action_prob * G
        loss = tf.keras.backend.mean(loss)

        return loss



