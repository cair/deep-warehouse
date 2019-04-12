import tensorflow as tf
from absl import logging


class Metrics:

    def __init__(self, agent):
        self.agent = agent
        self.episode = 1

        self.metrics = dict(
            reward=tf.keras.metrics.Sum(name="reward", dtype=self.agent.dtype),
            steps=tf.keras.metrics.Sum(name="steps", dtype=self.agent.dtype),
            total_loss=tf.keras.metrics.Mean(name="loss", dtype=self.agent.dtype),
            backprop_time=tf.keras.metrics.Mean(name="backprop_time", dtype=self.agent.dtype)
        )

    def reset(self):
        for k, metric in self.metrics.items():
            if isinstance(metric, tf.keras.metrics.Mean):
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
            metric_type = getattr(tf.keras.metrics, type)
            self.metrics[name] = metric_type(name, dtype=self.agent.dtype)

        self.metrics[name](value)

    def summary(self, name, data):
        tf.summary.scalar("sysx/%s" % name, data, self.episode)

    #def histogram(self, name, distribution):
    #    tf.scalar.histogram("sysx/%s" % name, distribution, self.episode)
