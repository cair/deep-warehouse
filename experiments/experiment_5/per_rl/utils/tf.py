import time
import numpy as np
import tensorflow as tf


def average_gradients(trainer_grads):

    average_grads = []
    for grads in zip(*trainer_grads):

        _grads = []
        for g in grads:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            _grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=_grads)
        grad = tf.reduce_mean(grad, 0)

        average_grads.append(grad)

    return average_grads


def average_weights(weights):
    average_weights = []

    for weight in list(zip(*weights)):
        average_weights.append(tf.reduce_mean(weight, axis=0))
    return average_weights
