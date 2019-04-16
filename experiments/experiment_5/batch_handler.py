import tensorflow as tf
import gym
import numpy as np
from absl import logging


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
        self._observations1 = []
        self._actions = []
        self._action_logits = []
        self._rewards = []
        self._terminals = []


        if self.episodic:
            self.minibatch_splits = 1
            if self.batch_sets != 1:
                logging.log(logging.WARN, "Batch mode is set to 'episodic' while batch_sets is not 1. Ignoring mini_batches config.")
        else:
            self.minibatch_splits = int((self.batch_size * self.batch_sets) / self.batch_size)

    def add(self, obs=None, obs1=None, action=None, action_logits=None, reward=None, terminal=None, increment=False):

        if obs is not None:
            self._observations.append(tf.cast(obs, dtype=self.dtype))
        if obs1 is not None:
            self._observations1.append(tf.cast(obs1, dtype=self.dtype))
        if action is not None:
            self._actions.append(tf.one_hot(action, self.action_space))
        if action_logits is not None:
            self._action_logits.append(tf.cast(action_logits, dtype=self.dtype))
        if reward is not None:
            self._rewards.append(tf.cast(reward, dtype=self.dtype))
        if terminal is not None:
            self._terminals.append(tf.cast(terminal, dtype=self.dtype))
            #if terminal:
                #self._terminal_indexes.append(len(self._terminals))

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
        self._observations1.clear()
        self._actions.clear()
        self._action_logits.clear()
        self._rewards.clear()
        self._terminals.clear()

    def obs(self):
        return tf.split(tf.concat(self._observations, axis=0), self.minibatch_splits, axis=0)

    def obs1(self):
        return tf.split(tf.concat(self._observations1, axis=0), self.minibatch_splits, axis=0)

    def act(self):
        return tf.split(tf.stack(self._actions), self.minibatch_splits)

    def act_logits(self):
        return tf.split(tf.concat(self._action_logits, axis=0), self.minibatch_splits)

    def rewards(self):
        return tf.split(tf.stack(self._rewards), self.minibatch_splits)

    def terminals(self):
        return tf.split(tf.stack(self._terminals), self.minibatch_splits)

    def get(self):
        return [self.obs(), self.obs1(), self.act(), self.act_logits(), self.rewards(), self.terminals()]


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
        self.b_obs1 = np.zeros(shape=(batch_size,) + obs_space.shape, dtype=dtype)
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
