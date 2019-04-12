import tensorflow as tf
import gym
import numpy as np


class BatchHandler:

    def __init__(self,
                 agent,
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 batch_size):
        self.agent = agent
        self.action_space = action_space
        self.episodic = self.agent.batch_mode == "episodic"

        # Convert dtype from tf to numpy
        if self.agent.dtype == tf.float32:
            dtype = np.float32
        elif self.agent.dtype == tf.float16:
            dtype = np.float16

        self.b_obs = np.zeros(shape=(batch_size, ) + obs_space.shape, dtype=dtype)
        self.b_obs1 = np.zeros(shape=(batch_size, ) + obs_space.shape, dtype=dtype)
        self.b_act = np.zeros(shape=(batch_size, action_space), dtype=dtype)
        self.b_act_logits = np.zeros(shape=(batch_size, action_space), dtype=dtype)
        self.b_rew = np.zeros(shape=(batch_size,), dtype=dtype)
        self.b_term = np.zeros(shape=(batch_size,), dtype=np.int8)

        self.batch_size = batch_size
        self.counter = 0
        self.last_counter = self.counter
        self.terminal_step_counter = 0

    def obs(self):
        return self.b_obs[:self.last_counter]

    def obs1(self):
        return self.b_obs1[:self.last_counter]

    def act(self):
        return self.b_act[:self.last_counter]

    def act_logits(self):
        return self.b_act_logits[:self.last_counter]

    def rewards(self):
        return self.b_rew[:self.last_counter]

    def terminals(self):
        return self.b_term[:self.last_counter]

    def add(self, obs=None, obs1=None, action=None, action_logits=None, reward=None, terminal=None, increment=False):

        """Increase size if full (for episodic)"""
        if self.episodic and int(len(self.b_obs)) == self.counter:
            self.b_obs = np.concatenate((self.b_obs, np.zeros_like(self.b_obs)), axis=0)
            self.b_obs1 = np.concatenate((self.b_obs1, np.zeros_like(self.b_obs1)), axis=0)
            self.b_act = np.concatenate((self.b_act, np.zeros_like(self.b_act)), axis=0)
            self.b_act_logits = np.concatenate((self.b_act_logits, np.zeros_like(self.b_act_logits)), axis=0)
            self.b_rew = np.concatenate((self.b_rew, np.zeros_like(self.b_rew)), axis=0)
            self.b_term = np.concatenate((self.b_term, np.zeros_like(self.b_term)), axis=0)

        if obs is not None:
            self.b_obs[self.counter] = obs
        if obs1 is not None:
            self.b_obs1[self.counter] = obs1
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
                self.terminal_step_counter = 0
            self.terminal_step_counter += 1

        if increment:

            if self.counter == self.batch_size - 1 and self.agent.batch_mode == "steps":
                """Stepwise batch increment."""
                self.last_counter = self.batch_size
                self.counter = 0
                return True

            elif bool(terminal) is True and self.agent.batch_mode == "episodic":
                """Episodic batch increment."""
                self.last_counter = self.counter
                self.counter = 0
                return True

            self.counter += 1
            return False




