import gym
import tensorflow as tf
import datetime
import os

from experiments.experiment_5.batch_handler import BatchHandler
from experiments.experiment_5.metrics import Metrics


class Agent:

    def __init__(self,
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 batch_size: int,
                 dtype,
                 policy,
                 tensorboard_enabled,
                 tensorboard_path,
                 optimizer=tf.keras.optimizers.Adam(lr=0.001)
                 ):
        self.loss_fns = {}
        self.action_space = action_space
        self.metrics = Metrics()
        self.policy = policy
        self.optimizer = optimizer
        self.name = self.__class__.__name__
        self.batch = BatchHandler(
            agent=self,
            obs_space=obs_space,
            action_space=action_space,
            batch_size=batch_size,
            dtype=dtype
        )

        if tensorboard_enabled:
            logdir = os.path.join(
                tensorboard_path,
                self.name + "_%s" % (datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
            )
            writer = tf.summary.create_file_writer(logdir)
            writer.set_as_default()

    def add_loss(self, name, lambda_fn):
        self.loss_fns[name] = lambda_fn

    def observe(self, reward, terminal):

        if terminal:
            self.metrics.episode += 1
            self.metrics.steps(self.metrics.steps_counter)
            self.metrics.steps_counter = 0
        else:
            self.metrics.steps_counter += 1

        return self.batch.add(
            reward=tf.cast(reward, tf.float32),
            terminal=tf.cast(terminal, tf.bool),
            increment=True
        )

    def train(self, observations):

        with tf.GradientTape() as tape:
            predicted_logits = self.policy(observations)

            total_loss = 0
            for loss_name, loss_fn in self.loss_fns.items():
                loss = loss_fn(predicted_logits)
                self.summary(loss_name, loss)
                total_loss += loss

        grads = tape.gradient(total_loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        self.summary("total_loss", total_loss)

        return total_loss

    def summary(self, name, data):
        tf.summary.scalar("%s/%s" % (self.name, name), data, self.optimizer.iterations)

