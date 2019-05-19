import time

import tensorflow as tf
import gym
import numpy as np
from absl import logging


class DynamicBatch:

    def __init__(self, agent, **kwargs):
        self.agent = agent
        self.episodic = self.agent.batch_mode == "episodic"
        self.batch_size = 1000000 if self.episodic else self.agent.batch_size
        self.n_mb = 1 if self.episodic else self.agent.mini_batches
        self.mb_size = self.batch_size // self.n_mb

        self.dtype = agent.dtype

        self.counter = 0
        self.data = {}  # Array of dicts.

    def add(self, **kwargs):

        for k, v in kwargs.items():
            try:
                self.data[k]
            except KeyError:

                self.data[k] = []

            self.data[k].append(np.squeeze(v))

        self.counter += 1
        if self.episodic:
            try:
                return bool(kwargs["terminal"])
            except KeyError:
                raise KeyError("In order to use episodic mode, 'terminal' key must be present in the dataset!")

        return self.counter == self.batch_size

    def get(self):
        return {k: np.asarray(v) for k, v in self.data.items()}

    def done(self):
        self.data = {k: [] for k in self.data.keys()}
        self.counter = 0
