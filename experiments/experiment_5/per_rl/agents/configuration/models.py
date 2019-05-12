import tensorflow as tf
from experiments.experiment_5.per_rl import utils

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
        self.logits = tf.keras.layers.Dense(self.agent.action_space, activation="linear", name='policy', dtype=self.agent.dtype)

    def shared(self, x):
        x = self.h_1(x)
        x = self.h_2(x)
        x = self.h_3(x)
        return x

    def call(self, inputs, **kwargs):
        x = self.shared(inputs)
        x = self.logits(x)
        return dict(
            logits=x
        )


class A2CPolicy(PGPolicy):
    """
    Nice resources:
    Blog: http://steven-anker.nl/blog/?p=184
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.h_4 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.h_5 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.h_6 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.action_value = tf.keras.layers.Dense(1, dtype=self.agent.dtype)

    def call(self, inputs, **kwargs):
        data = super().call(inputs, **kwargs)

        x = self.shared(inputs)
        x = self.h_4(x)
        x = self.h_5(x)
        x = self.h_6(x)
        action_value = self.action_value(x)

        data["action_value"] = tf.squeeze(action_value)
        return data


class PPOPolicy(Policy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.h_1 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.h_2 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.h_3 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.h_4 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)

        self.logits = tf.keras.layers.Dense(self.agent.action_space, activation="softmax", name='policy_logits',
                                            dtype=self.agent.dtype)

        self.action_value_1 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.action_value = tf.keras.layers.Dense(1,
                                                 activation="linear",
                                                 name="state_value",
                                                 dtype=self.agent.dtype
                                                 )

    def call(self, inputs, **kwargs):
        x = self.h_1(inputs)
        x = self.h_2(x)
        x = self.h_3(x)
        x = self.h_4(x)

        policy_logits = self.logits(x)
        action_value = self.action_value(self.action_value_1(x))

        return {
            "logits": policy_logits,
            "action_value": tf.squeeze(action_value)
        }
