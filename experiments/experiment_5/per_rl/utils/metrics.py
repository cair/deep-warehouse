from collections import deque
from absl import logging
import tensorflow as tf
import numpy as np



class Sum:

    def __init__(self):
        self.total = 0
        self.total_total = 0

    def __call__(self, v):
        self.total += v
        self.total_total += v

    def results(self, total=False):
        if total:
            return self.total_total
        else:
            return self.total

    def reset_states(self):
        self.total = 0


class Mean:

    def __init__(self, sum_at_reset=False):
        self.data = []
        self.series = deque(maxlen=10)
        self.series_total = []
        self.op = np.mean
        self.reset_op = np.sum if sum_at_reset else np.mean

    def __call__(self, v):
        self.data.append(v)

    def results(self, total=False):

        if total and len(self.series_total) > 0:
            return self.op(self.series_total)
        elif not total and len(self.series) > 0:
            return self.op(self.series)
        else:
            return self.op(self.data)

    def reset_states(self):
        calc = self.reset_op(self.data)
        self.series.append(calc)
        self.series_total.append(calc)
        self.data = []


class Stddev(Mean):

    def __init__(self):
        super().__init__()
        self.op = np.std


class GenericMetric:

    def __init__(self,
                 engine,
                 name,
                 type,
                 tags,
                 episode,
                 epoch,
                 total
                 ):
        self.engine = engine
        self.name = name
        self.tags = "" if tags is None else tags + "/"
        self.type = type
        self.episode = episode
        self.epoch = epoch
        self.total = total

        self.data_types = dict(
            mean=Mean,
            sum=Sum,
            sum_mean=lambda: Mean(sum_at_reset=True)
        )

        self.data = dict(
            episode=self.data_types[type](),
            epoch=self.data_types[type](),
        )

    def __call__(self, a):
        if self.episode:
            self.data["episode"](a)
        if self.epoch:
            self.data["epoch"](a)

    def done(self, episode=False, epoch=False):

        types = []
        if self.episode and episode:
            types.append("episode")

        if self.epoch and epoch:
            types.append("epoch")

        for type in types:

            data = self.data[type]
            self.engine.summary(self.fullname(type), data.results(), measure=type)
            data.reset_states()

            if self.total:
                self.engine.summary(self.fullname(type + "_total"), data.results(total=True), measure=type)

    @property
    def tagname(self):
        return self.tags + self.name

    def fullname(self, t):
        return self.tagname + "/" + self.type + "/" + t


class Metrics:
    supported = {
        "mean": "Between last metric and current metric",
        "sum": "Sum of all collected data",
        "sum_mean": "Sums then averages instead of mean of means.",
        "stddev": "Standard deviation between the collected data"
    }

    def __init__(self, agent):
        self.agent = agent

        self.measures = dict(
            episode=0,
            epoch=0,
        )

        self.metrics = {
            t: {} for t in Metrics.supported.keys()
        }

        self.summary_data = {
            t: {} for t in self.measures.keys()
        }

        for k, v in Metrics.supported.items():
            self.text("Metrics", k + ": " + v)

    def reset(self):
        for metric in self.metrics:
            metric.reset_states()

    def summarize(self, items, episode=True, epoch=False):
        if episode and epoch:
            raise OverflowError("You cannot issue summarization for both episode and epoch at the same time!")

        summary_category = None
        if episode:
            summary_category = "episode"
        if epoch:
            summary_category = "epoch"

        logging.log(logging.DEBUG, " | ".join(["%s: %s" % (k, v) for k, v in self.summary_data[summary_category].items()]))

    def get(self, name):
        return self.metrics[name]

    def add(self, name, value, types, tags, episode=False, epoch=False, total=False):

        # There is no metric of that name
        for type in types:
            if type not in Metrics.supported:
                raise NotImplementedError("Metric of type %s is not supported." %type)

            if name not in self.metrics[type]:
                self.metrics[type][name] = GenericMetric(
                    self,
                    name=name,
                    type=type,
                    tags=tags,
                    episode=episode,
                    epoch=epoch,
                    total=total
                )

            self.metrics[type][name](float(value))

    def done(self, episode=False, epoch=False):
        if episode:
            self.measures["episode"] += 1

        if epoch:
            self.measures["epoch"] += 1

        for type, metrics in self.metrics.items():
            for name, metric in metrics.items():
                metric.done(episode=episode, epoch=epoch)

    def summary(self, name, data, measure):
        try:
            measure_val = self.measures[measure]
            self.summary_data[measure][name] = data
            tf.summary.scalar("%s" % name, data, measure_val)
        except KeyError:
            raise KeyError("Could not find the measure type %s in supported measures!" % measure)

    def text(self, name, data):
        tf.summary.text(name, data, 0)


    # def histogram(self, name, distribution):
    #    tf.scalar.histogram("sysx/%s" % name, distribution, self.episode)
