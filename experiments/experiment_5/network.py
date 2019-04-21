import time

import tensorflow as tf
import tensorflow_probability as tfp
from experiments.experiment_5 import utils


class Policy(tf.keras.models.Model):

    DEFAULTS = dict()
    arguments = utils.arguments

    def __init__(self, agent,
                 inference,
                 training,
                 optimizer=tf.keras.optimizers.Adam(lr=0.001)
                 ):
        super(Policy, self).__init__()
        self.agent = agent
        self.inference = inference
        self.training = training
        self.optimizer = optimizer


class PGPolicy(Policy):

    DEFAULTS = dict()

    def __init__(self, **kwargs):
        super(PGPolicy, self).__init__(**kwargs)

        self.h_1 = tf.keras.layers.Dense(64, activation='relu', dtype=self.agent.dtype, kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.h_2 = tf.keras.layers.Dense(64, activation='relu', dtype=self.agent.dtype, kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.h_3 = tf.keras.layers.Dense(64, activation='relu', dtype=self.agent.dtype, kernel_initializer=tf.keras.initializers.glorot_uniform())

        # Probabilties of each action
        self.logits = tf.keras.layers.Dense(self.agent.action_space, activation="linear", name='policy_logits', dtype=self.agent.dtype)

    def shared(self, x):
        x = self.h_1(x)
        x = self.h_2(x)
        x = self.h_3(x)
        return x

    def call(self, inputs):
        x = self.shared(inputs)
        y = self.logits(x)

        x = tfp.distributions.Categorical(logits=y)

        return dict(
            policy_logits=y,
            policy_sample=tf.squeeze(x.sample())
        )
