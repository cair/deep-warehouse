import tensorflow as tf


class PGPolicy(tf.keras.models.Model):

    def __init__(self, action_space, dtype=tf.float32):
        super(PGPolicy, self).__init__()

        self._dtype = dtype
        self.training = True
        self.action_space = action_space

        self.h_1 = tf.keras.layers.Dense(64, activation='relu', dtype=self._dtype, kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.h_2 = tf.keras.layers.Dense(64, activation='relu', dtype=self._dtype, kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.h_3 = tf.keras.layers.Dense(64, activation='relu', dtype=self._dtype, kernel_initializer=tf.keras.initializers.glorot_uniform())

        # Probabilties of each action
        self.logits = tf.keras.layers.Dense(action_space, activation="softmax", name='policy_logits', dtype=self._dtype)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=self._dtype)
        x = self.h_1(x)
        x = self.h_2(x)
        x = self.h_3(x)
        return self.logits(x)
