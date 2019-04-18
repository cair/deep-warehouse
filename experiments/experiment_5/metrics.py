import sys
from collections import deque

import tensorflow as tf
from absl import logging
import numpy as np


class Sum(list):

    def __init__(self, name, dtype, **kwargs):
        super().__init__()

    def __call__(self, a):
        self.append(a)

    def reset_states(self):
        self.clear()

    def result(self):
        return np.sum(self)


class Mean(deque):
    def __init__(self, name, dtype, maxlen):
        super().__init__(maxlen=maxlen)

    def __call__(self, a):
        self.append(a)

    def reset_states(self):
        self.clear()

    def result(self):
        if len(self) == 0:
            return 0

        return np.nanmean(self)


class InfiniteMean(Mean):

    def __init__(self, name, dtype, maxlen):
        super().__init__(name, dtype, maxlen=None)


class EpisodicMean(Mean):
    pass


class Metrics:

    def __init__(self, agent):
        self.agent = agent
        self.episode = 1
        self.avg_len = 100

        self.metrics = dict(
            reward=Sum(name="reward", dtype=self.agent.dtype),
            steps=Sum(name="steps", dtype=self.agent.dtype),
            total_loss=Mean(name="loss", dtype=self.agent.dtype, maxlen=self.avg_len),
            backprop_time=Mean(name="backprop_time", dtype=self.agent.dtype, maxlen=self.avg_len)
        )

    def reset(self):
        for k, metric in self.metrics.items():
            if isinstance(metric, Mean):
                continue

            metric.reset_states()

    def get(self, name):
        return self.metrics[name]

    def summarize(self):
        res_str = "Episode: %s | " % self.episode
        for k, metric in self.metrics.items():
            r = metric.result()
            res_str += "%s: %.3f | " % (k.capitalize(), float(r))
            self.summary(k, r)
        logging.log(logging.DEBUG, res_str)
        self.reset()
        self.episode += 1

    def add(self, name, value, type="Mean"):
        if name not in self.metrics:
            metric_type = getattr(sys.modules[__name__], type)
            self.metrics[name] = metric_type(name, dtype=self.agent.dtype, maxlen=self.avg_len)

        self.metrics[name](value)

    def summary(self, name, data):
        tf.summary.scalar("sysx/%s" % name, data, self.episode)

    def text(self, name, data):
        tf.summary.text(name, data, self.episode)

    #def histogram(self, name, distribution):
    #    tf.scalar.histogram("sysx/%s" % name, distribution, self.episode)
