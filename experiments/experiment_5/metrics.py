import tensorflow as tf


class Metrics:

    def __init__(self):
        self.episode = 0
        self.total_loss = tf.keras.metrics.Mean(name="total_loss", dtype=tf.float32)
        self.training_time = tf.keras.metrics.Mean(name="training_time", dtype=tf.float32)
        self.steps_counter = 0
        self.steps = tf.keras.metrics.Mean(name="steps", dtype=tf.float32)

