import sys
import time
from collections import deque

import tensorflow as tf
from absl import logging
import numpy as np


class Sum(list):
    name = "/sum"

    def __init__(self, name, dtype, **kwargs):
        super().__init__()

    def __call__(self, a):
        self.append(a)

    def reset_states(self):
        self.clear()

    def result(self):
        return np.sum(self)


class SumInfinite:
    name = "/sum_total"

    def __init__(self, name, dtype, **kwargs):
        super().__init__()
        self.c = 0

    def __call__(self, a):
        self.c += a

    def reset_states(self):
        pass

    def result(self):
        return self.c


class Mean(deque):
    name = "/mean"

    def __init__(self, name, dtype, maxlen):
        super().__init__(maxlen=maxlen)
        self.name = name

    def __call__(self, a):
        self.append(a)

    def reset_states(self):
        self.clear()

    def result(self):
        if len(self) == 0:
            return 0

        return np.mean(self)


class MeanInfinite(Mean):
    name = "/mean_total"

    def __init__(self, name, dtype, maxlen):
        super().__init__(name, dtype, maxlen=None)

    def reset_states(self):
        pass


class GenericMetric:

    def __init__(self, name, dtype, types, tags):
        self.old_data = []
        self.old_data_summed = []
        self.data = []
        self.name = name
        self.tags = tags
        self.types = types

        self.fns = {
            "mean_episode": lambda: np.mean(self.data, dtype=np.float32),
            "sum_mean_frequent": lambda: np.mean(self.old_data_summed[-100:]),
            "sum_mean_total": lambda: np.mean(self.old_data_summed),
            "mean_total": lambda: np.mean(self.old_data + self.data),
            "sum_episode": lambda: np.sum(self.data),
            "sum_total": lambda: np.sum(self.old_data + self.data),
            "stddev_episode": lambda: np.std(self.data),
            "stddev_total": lambda: np.std(self.old_data + self.data)
        }

        # Sum Total (All episodes)
        # Sum (One episode)
        # Average (Episode)
        # Average (All Episodes)

    def __call__(self, a):

        self.data.append(a)

    def reset_states(self):
        self.old_data_summed.append(np.sum(self.data))
        self.old_data.extend(self.data)
        self.data.clear()

    def result(self):
        return {
            d: self.fns[d]() for d in self.types
        }

    @property
    def tagname(self):
        return self.tags + "/" + self.name

    def fullname(self, metric):
        return self.tagname + "/" + metric


class Metrics:

    def __init__(self, agent):
        self.agent = agent
        self.episode = 0
        self.epoch = 1
        self.avg_len = 100
        self.metrics = {}

        self.fns_explaination = {
            "mean_episode": "After an episode is completed, this take the mean over all elements collected",
            "sum_mean_frequent": "After an episode is completed, this take the mean over summed values from current and previous episodes",
            "mean_total": "Takes the mean over all elements collected so far.",
            "sum_episode": "Sums all the values collected for the episode",
            "sum_total": "Sums all the values collected for all episodes",
            "stddev_episode": "Standard deviation for data collected in a single episode",
            "stddev_total": "Standard deviation for data collected in all episodes"
        }

        for k, v in self.fns_explaination.items():
            self.text("Metrics", k + ": " + v)


    def reset(self):
        for k, metric in self.metrics.items():
            metric.reset_states()

    def get(self, name):
        return self.metrics[name]

    def summarize(self):
        res_str = "Episode: %s | " % self.episode
        for k, metric in self.metrics.items():

            data = metric.result()
            for k, r in data.items():
                k = metric.fullname(k)

                res_str += "%s: %.5f | " % (k.capitalize(), float(r))
                if not np.isnan(r):
                    self.summary(k, r)

        logging.log(logging.DEBUG, res_str)
        self.reset()
        self.episode += 1

    def add(self, name, value, types, tags):

            if name not in self.metrics:
                self.metrics[name] = GenericMetric(
                    name=name,
                    dtype=self.agent.dtype,
                    types=types,
                    tags=tags
                )

            self.metrics[name](value)

    def summary(self, name, data):
        tf.summary.scalar("%s" % name, data, self.episode)

    def text(self, name, data):
        tf.summary.text(name, data, self.episode)

    # def histogram(self, name, distribution):
    #    tf.scalar.histogram("sysx/%s" % name, distribution, self.episode)
