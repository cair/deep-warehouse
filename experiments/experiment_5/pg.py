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


class PGAgent(Agent):

    def __init__(self,
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 gamma=0.99,
                 batch_size=1,
                 dtype=tf.float32,
                 tensorboard_enabled=True,
                 tensorboard_path="./tb/",
                 ):

        super(PGAgent, self).__init__(
            obs_space=obs_space,
            action_space=action_space,
            batch_size=batch_size,
            dtype=dtype,
            policy=PGPolicy(
                action_space=action_space,
                dtype=dtype
            ),
            tensorboard_enabled=tensorboard_enabled,
            tensorboard_path=tensorboard_path
        )
        self.gamma = gamma

        self.add_loss("policy_loss", lambda y: self.policy_loss(self.batch.b_act, self.G(self.batch.b_rew, self.batch.b_term), y))

    def reset(self):
        self.batch.counter = 0

    def predict(self, observation):
        action_logits = self.policy.predict(observation)

        action_sample = np.random.choice(np.arange(self.action_space), p=tf.squeeze(action_logits).numpy())

        self.batch.add(obs=observation, action_logits=action_logits, action=action_sample)
        return action_sample

    def observe(self, reward, terminal):
        super().observe(reward, terminal)

        if self.batch.counter == 0:
            s = time.time()
            loss = self.train(self.batch.b_obs)
            self.metrics.total_loss(loss)
            self.metrics.training_time(time.time() - s)

            logging.debug("Episode %d | Avg_Steps: %f | Training loss %f | Time Elapsed: %f",
                          self.metrics.episode,
                          self.metrics.steps.result(),
                          self.metrics.total_loss.result(),
                          self.metrics.training_time.result())

            self.summary("training_time", self.metrics.training_time.result())
            self.summary("training_loss_smooth", self.metrics.total_loss.result())

    """
    # G is commonly refered to as the cumulative discounted rewards
    """

    def G(self, rewards, terminals):
        """ take 1D float array of rewards and compute discounted reward """

        discounted_rewards = np.zeros_like(rewards)

        cum_r = 0
        for i in reversed(range(0, len(rewards))):
            cum_r = rewards[i] + (cum_r * self.gamma) * (1 - terminals[i])

            discounted_rewards[i] = cum_r

        discounted_rewards = (discounted_rewards - discounted_rewards.std()) / discounted_rewards.mean()

        return discounted_rewards

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


