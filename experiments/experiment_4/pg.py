import time

import tensorflow as tf
import gym
import numpy as np
from scipy import signal


class BatchHandler:

    def __init__(self,
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 batch_size,
                 dtype=tf.float32):

        self.action_space = action_space

        # Convert dtype from tf to numpy
        if dtype == tf.float32:
            dtype = np.float32
        elif dtype == tf.float16:
            dtype = np.float16

        self.b_obs = np.zeros(shape=(batch_size, ) + obs_space.shape, dtype=dtype)
        self.b_act = np.zeros(shape=(batch_size,), dtype=dtype)
        self.b_act_logits = np.zeros(shape=(batch_size, action_space), dtype=dtype)
        self.b_rew = np.zeros(shape=(batch_size,), dtype=dtype)
        self.b_term = np.zeros(shape=(batch_size,), dtype=np.int8)

        self.batch_size = batch_size
        self.counter = 0

    def add(self, obs=None, action=None, action_logits=None, reward=None, terminal=None, increment=False):

        if obs is not None:
            self.b_obs[self.counter] = obs
        if action is not None:
            self.b_act[self.counter] = action
        if action_logits is not None:
            self.b_act_logits[self.counter] = action_logits
        if reward is not None:
            self.b_rew[self.counter] = reward
        if terminal is not None:
            self.b_term[self.counter] = terminal

        if increment:
            if self.counter == self.batch_size - 1:
                self.counter = 0
                return True

            self.counter += 1
            return False


class PGPolicy(tf.keras.models.Model):

    def __init__(self, action_space: gym.spaces.Discrete, dtype=tf.float32):
        super(PGPolicy, self).__init__()
        self._dtype = dtype
        self.training = True
        self.action_space = action_space

        self.h_1 = tf.keras.layers.Dense(128, activation='relu', dtype=self._dtype)
        self.h_2 = tf.keras.layers.Dense(128, activation='relu', dtype=self._dtype)
        self.h_3 = tf.keras.layers.Dense(128, activation='relu', dtype=self._dtype)

        # Probabilties of each action
        self.logits = tf.keras.layers.Dense(action_space, activation="softmax", name='policy_logits', dtype=self._dtype)

        self.optimizer = tf.keras.optimizers.Adam(lr=0.0001)  # TODO dynamic

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=self._dtype)
        x = self.h_1(x)
        x = self.h_2(x)
        x = self.h_3(x)
        return self.logits(x)

    def loss_logits(self, y_act_actual, y_adv_actual, y_pred):
        # http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/

        #true_y = actions * advantages

        #log_likelihood = tf.math.log(true_y * (true_y - pred_y) + (1 - true_y) * (true_y + pred_y))

        #loss = tf.reduce_mean(log_likelihood)

        #actions = tf.cast(actions, tf.int32)
        #weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        #loss = weighted_sparse_ce(actions, pred_y, sample_weight=advantages)



        return 0 # loss # - 0.001 * entropy_loss

    def train(self, observations, actions, actions_logits, discounted_rewards):

        with tf.GradientTape() as tape:

            predicted_logits = self(observations)

            actions_onehot = tf.keras.utils.to_categorical(actions, num_classes=self.action_space)
            log_action_prob = tf.keras.backend.log(
                tf.keras.backend.sum(predicted_logits * actions_onehot, axis=1)
            )
            loss = - log_action_prob * discounted_rewards
            loss = tf.keras.backend.mean(loss)


            grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss.numpy()


class PGAgent:

    def __init__(self,
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 gamma=0.99,
                 batch_size=1,
                 dtype=tf.float32
                 ):

        self.gamma = gamma

        self.action_space = action_space

        self.policy = PGPolicy(
            action_space=action_space,
            dtype=dtype
        )

        self.batch = BatchHandler(
            obs_space=obs_space,
            action_space=action_space,
            batch_size=batch_size,
            dtype=dtype
        )
        self.pred_steps = 0

    def reset(self):
        self.batch.counter = 0

    def predict(self, observation):
        action_logits = self.policy.predict(observation)

        #action_sample = tf.argmax(tf.squeeze(action_logits)).numpy()
        #action_sample = tf.squeeze(tf.random.categorical(action_logits, 1)).numpy()

        action_sample = np.random.choice(np.arange(self.action_space), p=tf.squeeze(action_logits).numpy())

        self.batch.add(obs=observation, action_logits=action_logits, action=action_sample)
        self.pred_steps += 1
        return action_sample

    def observe(self, reward, terminal):

        do_train = self.batch.add(reward=reward, terminal=terminal, increment=True)

        # Batch is full.
        if do_train:
            observations = self.batch.b_obs
            actions = self.batch.b_act
            actions_logits = self.batch.b_act_logits
            rewards = self._discounted_rewards(self.batch.b_rew)
            losses = self.policy.train(observations, actions, actions_logits, rewards)
            return losses

    def _discounted_rewards(self, rewards):
        """ take 1D float array of rewards and compute discounted reward """
        r = rewards[::-1]
        a = [1, -self.gamma]
        b = [1]
        y = np.flip(signal.lfilter(b, a, x=r))

        # https://statisticsbyjim.com/glossary/standardization/
        # Also: https://en.wikipedia.org/wiki/Feature_scaling
        #y = (y - y.std()) / y.mean()

        # Baseline:
        # http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_4_policy_gradient.pdf
        # Read above. migh be nice ^
        # https://stats.stackexchange.com/questions/357519/what-is-a-baseline-function-in-policy-gradients-methods

        return y - rewards.mean()

env = gym.make('CartPole-v0')
agent = PGAgent(
    obs_space=env.observation_space,
    action_space=env.action_space.n,
    batch_size=128
)



for e in range(30000):

    steps = 0
    terminal = False
    obs = env.reset()
    cum_loss = 0
    loss_n = 0

    while not terminal:

        action = agent.predict(obs[None, :])

        obs, reward, terminal, info = env.step(action)
        reward = 0 if terminal else reward

        losses = agent.observe(reward, terminal)
        if losses is not None:
            loss_n += 1
            cum_loss += losses
        steps += 1

    if loss_n != 0:
        print("E: %s, Steps %s, Loss: %s" % (e, steps, cum_loss / (0.00000001 + loss_n)))
