import tensorflow as tf
from absl import logging


class Metrics:

    def __init__(self, agent):
        self.agent = agent
        self.episode = 1

        self.metrics = dict(
            reward=tf.keras.metrics.Sum(name="reward", dtype=tf.float32),
            steps=tf.keras.metrics.Sum(name="steps", dtype=tf.float32),
            total_loss=tf.keras.metrics.Mean(name="loss", dtype=tf.float32),
            backprop_time=tf.keras.metrics.Mean(name="backprop_time", dtype=tf.float32)
        )

    def reset(self):
        for k, metric in self.metrics.items():
            if isinstance(metric, tf.keras.metrics.Mean):
                continue
            metric.reset_states()
        
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
            self.metrics[name] = metric_type(name, dtype=tf.float32)

        self.metrics[name](value)

    def summary(self, name, data):
        tf.summary.scalar("%s/%s" % (self.agent.name, name), data, self.episode)
