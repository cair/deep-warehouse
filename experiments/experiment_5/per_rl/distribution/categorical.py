import tensorflow as tf


class Categorical:

    @staticmethod
    def logp(logits):
        p = tf.maximum(x=tf.math.softmax(logits=logits, axis=-1), y=1e-6)
        logp = tf.math.log(x=p)
        return logp

    @staticmethod
    def logpac(logits, actions):
        return tf.reduce_sum(tf.math.log_softmax(logits, axis=1) * actions, axis=1)

    @staticmethod
    def neglogpac(logits, actions):
        return -Categorical.logpac(logits, actions)

    @staticmethod
    def sample(logits):
        return tf.squeeze(tf.random.categorical(logits, 1))

    @staticmethod
    def entropy(logits):
        logp = tf.math.softmax(logits)
        return -tf.reduce_sum(logp * tf.math.log(logp + 1e-10), axis=1)
