import tensorflow as tf
import gym
import numpy as np


class BatchHandler:

    def __init__(self,
                 agent,
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 batch_size,
                 dtype=tf.float32):
        self.agent = agent
        self.action_space = action_space

        # Convert dtype from tf to numpy
        if dtype == tf.float32:
            dtype = np.float32
        elif dtype == tf.float16:
            dtype = np.float16

        self.b_obs = np.zeros(shape=(batch_size, ) + obs_space.shape, dtype=dtype)
        self.b_act = np.zeros(shape=(batch_size, action_space), dtype=dtype)
        self.b_act_logits = np.zeros(shape=(batch_size, action_space), dtype=dtype)
        self.b_rew = np.zeros(shape=(batch_size,), dtype=dtype)
        self.b_term = np.zeros(shape=(batch_size,), dtype=np.int8)

        self.batch_size = batch_size
        self.counter = 0
        self.terminal_step_counter = 0

    def add(self, obs=None, action=None, action_logits=None, reward=None, terminal=None, increment=False):

        if obs is not None:
            self.b_obs[self.counter] = obs
        if action is not None:
            self.b_act[self.counter][:] = 0
            self.b_act[self.counter][action] = 1
        if action_logits is not None:
            self.b_act_logits[self.counter] = action_logits
        if reward is not None:
            self.b_rew[self.counter] = reward
        if terminal is not None:
            self.b_term[self.counter] = terminal
            if terminal:
                self.agent.summary("steps", self.terminal_step_counter)
                self.terminal_step_counter = 0
            self.terminal_step_counter += 1

        if increment:
            if self.counter == self.batch_size - 1:
                self.counter = 0
                return True

            self.counter += 1
            return False