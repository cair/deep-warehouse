import time

import tensorflow as tf
import gym
import numpy as np
from absl import logging


class DynamicBatch:

    def __init__(self, agent, **kwargs):
        self.agent = agent
        self.episodic = self.agent.batch_mode == "episodic"
        self.bsize = self.agent.batch_size
        self.mb = self.agent.mini_batches

        self.dtype = agent.dtype

        self.counter = 0
        self.data = dict()

        if self.episodic:
            self.mb_count = 1
            if self.mb != 1:
                logging.log(logging.WARN, "Batch mode is set to 'episodic' while batch_sets is not 1. Ignoring "
                                          "mini_batches config.")
        else:
            self.mb_count = int((self.bsize * self.mb) / self.bsize)

        self.total_size = self.bsize * self.mb

    def add(self, **kwargs):

        for k, v in kwargs.items():
            try:
                data_container = self.data[k]
            except KeyError:
                self.data[k] = []
                data_container = self.data[k]

            # Convert to numpy
            v = np.squeeze(np.asarray(v))

            #if v.ndim == 0:
            #    v = np.reshape(v, v.shape + (1, ))

            data_container.append(np.squeeze(v))
        self.counter += 1

        if self.episodic:
            try:
                return bool(kwargs["terminal"])

            except KeyError:
                raise KeyError("In order to use episodic mode, 'terminal' key must be present in the dataset!")

        return self.counter == self.total_size

    def flush(self):
        """data = dict()
        for k in list(self.data.keys()):
            data[k] = np.asarray(
                np.split(
                    np.asarray(self.data[k], dtype=self.dtype.name),
                    self.mb_count,
                    axis=0
                ))
            del self.data[k]
        return data"""

        # TODO typical optimization point
        data = [{} for _ in range(self.mb_count)]
        for k in list(self.data.keys()):
            for i, elem in enumerate(np.asarray(
                np.split(
                    np.asarray(self.data[k], dtype=self.dtype.name),
                    self.mb_count,
                    axis=0
                ))):
                data[i][k] = elem

            del self.data[k]
        return data



class VectorBatchHandler:

    def __init__(self, agent,
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 batch_size):
        self.agent = agent
        self.action_space = action_space
        self.episodic = self.agent.batch_mode == "episodic"
        self.batch_size = batch_size
        self.batch_sets = self.agent.mini_batches
        self.dtype = agent.dtype
        self.counter = 0

        # Lists
        self._observations = []
        self.state = None # Current state (latest)
        self._actions = []
        self._rewards = []
        self._terminals = []
        self._values = []

        if self.episodic:
            self.minibatch_splits = 1
            if self.batch_sets != 1:
                logging.log(logging.WARN, "Batch mode is set to 'episodic' while batch_sets is not 1. Ignoring mini_batches config.")
        else:
            self.minibatch_splits = int((self.batch_size * self.batch_sets) / self.batch_size)

    def add(self, obs=None, obs1=None, action=None, reward=None, terminal=None, policy_sample=None, action_value=None, increment=False, **kwargs):

        if obs is not None:
            self._observations.append(np.squeeze(obs))
        if obs1 is not None:
            self.state = [[obs1]]  # Current state (latest)
        if policy_sample is not None:
            self._actions.append(tf.one_hot(policy_sample, self.action_space))
        if action_value is not None:
            self._values.append(np.expand_dims(np.squeeze(action_value, 1), 1))
        if reward is not None:
            self._rewards.append([reward])
        if terminal is not None:
            self._terminals.append([float(terminal)])

        if increment:

            if self.agent.batch_mode == "steps":
                """Stepwise batch increment."""

                counter_is_met = (self.batch_size * self.batch_sets) - 1 == self.counter

                if counter_is_met:
                    self.counter = 0
                    return True
            elif bool(terminal) is True and self.agent.batch_mode == "episodic":
                """Episodic batch increment."""
                self.counter = 0
                return True

            self.counter += 1
            return False

    def clear(self):
        self._observations.clear()
        self.state = None
        self._actions.clear()

        self._rewards.clear()
        self._terminals.clear()

    def obs(self):

        return np.asarray(np.vsplit(np.asarray(self._observations, dtype=self.dtype.name), self.minibatch_splits))

    def obs1(self):
        return np.asarray(self.state, dtype=self.dtype.name)

    def act(self):
        return np.asarray(np.vsplit(np.asarray(self._actions, dtype=self.dtype.name), self.minibatch_splits))

    def rewards(self):
        return np.asarray(np.vsplit(np.asarray(self._rewards, dtype=self.dtype.name), self.minibatch_splits)) # # .swapaxes(1, 0)

    def terminals(self):
        return np.asarray(np.vsplit(np.asarray(self._terminals, dtype=self.dtype.name), self.minibatch_splits))

    def values(self):
        print(np.asarray(self._values, dtype=self.dtype.name).shape, "lal")
        return np.asarray(np.vsplit(np.asarray(self._values, dtype=self.dtype.name), self.minibatch_splits))

    def get(self):
        return [self.obs(), self.obs1(), self.act(), self.values(), self.rewards(), self.terminals()]


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

        self.b_obs = np.zeros(shape=(batch_size,) + obs_space.shape, dtype=dtype)
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
