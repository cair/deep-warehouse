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
                      lambda prediction, data: self.policy_loss(
                          data["actions"],
                          self.discounted_returns(
                              data["rewards"],
                              data["terminals"]
                          ),
                          prediction["policy_logits"]
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

    def discounted_returns(self, rewards, terminals):
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

    def GAE(self, values, next_value, rewards, terminals, gamma=0.99, tau=0.95):
        values = np.concatenate((values.numpy(), [next_value]))
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):

            delta = rewards[step] + gamma * values[step + 1] * terminals[step] - values[step]
            gae = delta + gamma * tau * terminals[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    """
    A: one hot vectors of actions
    G: discounted rewards (advantages)
    predicted_logits: The network's predicted logits.
    """
    def policy_loss(self, A, G, predicted_logits):

        neg_log_policy = -tf.math.log(tf.clip_by_value(predicted_logits, 1e-7, 1))
        loss = tf.reduce_mean(tf.reduce_sum(neg_log_policy * A, axis=1) * G)

        #log_action_prob = tf.keras.backend.log(
        #    tf.keras.backend.sum(predicted_logits * A, axis=1)
        #)

        #loss = - log_action_prob * G
        #loss = tf.keras.backend.mean(loss)
        return loss



