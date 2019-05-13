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

    def __init__(self,
                 engine,
                 name,
                 dtype,
                 type,
                 tags,
                 episode,
                 epoch,
                 total
                 ):
        self.engine = engine
        self.name = name
        self.tags = tags
        self.type = type
        self.episode = episode
        self.epoch = epoch
        self.total = total

      

        self.calculated_episode = 0
        self.calculated_epoch = 0
        self.calculated_total = 0

        self.data = []

        """self.old_data = []
        self.old_data_summed = []
        self.data = []
 

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
        # Average (All Episodes)"""

    def __call__(self, a):
        self.data.append(a)

    def update(self):
        print(self.episode, self.epoch)

        if self.episode:



    def result(self):
        #if self.episode:
        return {
            "test": 1
            #d: self.fns[d]() for d in self.types
        }

    @property
    def tagname(self):
        return self.tags + "/" + self.name

    def fullname(self, metric):
        return self.tagname + "/" + metric


class Metrics:
    supported = {
        "mean": "Between last metric and current metric",
        "sum": "",
        # "mean_episode": "After an episode is completed, this take the mean over all elements collected",
        # "sum_mean_frequent": "After an episode is completed, this take the mean over summed values from current and previous episodes",
        # "mean_total": "Takes the mean over all elements collected so far.",
        # "sum_episode": "Sums all the values collected for the episode",
        # "sum_total": "Sums all the values collected for all episodes",
        # "stddev_episode": "Standard deviation for data collected in a single episode",
        # "stddev_total": "Standard deviation for data collected in all episodes"
    }

    def __init__(self, agent):
        self.agent = agent

        self.measures = dict(
            episode=1,
            epoch=1,
        )
        self.metrics = {
            t: {} for t in Metrics.supported.keys()
        }

        for k, v in Metrics.supported.items():
            self.text("Metrics", k + ": " + v)

    def reset(self):
        for metric in self.metrics:
            metric.reset_states()

    def get(self, name):
        return self.metrics[name]

    def summarize(self, stdout=False):

        if stdout:
            stdout_vec = [
                "%s: %s" % (k.capitalize(), v) for k, v in self.measures.items()
            ]

        for type, metrics in self.metrics.items():
            for name, metric in metrics.items():

                for k, r in metric.result().items():
                    k = metric.fullname(k)



                    if stdout:
                        stdout_vec.append("%s: %.5f | " % (k.capitalize(), float(r)))

                    if not np.isnan(r):
                        self.summary(k, r)

            logging.log(logging.DEBUG, " | ".join(stdout_vec))

    def add(self, name, value, types, tags, episode=False, epoch=False, total=False):

        # There is no metric of that name
        for type in types:
            if type not in Metrics.supported:
                raise NotImplementedError("Metric of type %s is not supported." %type)

            if name not in self.metrics:
                self.metrics[type][name] = GenericMetric(
                    self,
                    name=name,
                    dtype=self.agent.dtype,
                    type=type,
                    tags=tags,
                    episode=episode,
                    epoch=epoch,
                    total=total
                )

            self.metrics[type][name](value)

    def done(self, episode=False, epoch=False):
        if episode:
            self.measures["episode"] += 1

        if epoch:
            self.measures["epoch"] += 1

        for type, metrics in self.metrics.items():
            for name, metric in metrics.items():
                metric.update()

    def summary(self, name, data):
        tf.summary.scalar("%s" % name, data, 0)

    def text(self, name, data):
        tf.summary.text(name, data, 0)


    # def histogram(self, name, distribution):
    #    tf.scalar.histogram("sysx/%s" % name, distribution, self.episode)
