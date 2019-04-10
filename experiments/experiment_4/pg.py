import datetime
import os

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import gym
import numpy as np
from absl import flags, logging
import time
FLAGS = flags
logging.set_verbosity(logging.DEBUG)


class BatchHandler:

    def __init__(self,
                 agent,
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 batch_size,
                 dtype=tf.float32):
        self.agent = agent
        self.action_space = action_space

        # Convert dtype from tf to numpy
        if dtype == tf.float32:
            dtype = np.float32
        elif dtype == tf.float16:
            dtype = np.float16

        self.b_obs = np.zeros(shape=(batch_size, ) + obs_space.shape, dtype=dtype)
        self.b_act = np.zeros(shape=(batch_size, action_space), dtype=dtype)
        self.b_act_logits = np.zeros(shape=(batch_size, action_space), dtype=dtype)
        self.b_rew = np.zeros(shape=(batch_size,), dtype=dtype)
        self.b_term = np.zeros(shape=(batch_size,), dtype=np.int8)

        self.batch_size = batch_size
        self.counter = 0
        self.terminal_step_counter = 0

    def add(self, obs=None, action=None, action_logits=None, reward=None, terminal=None, increment=False):

        if obs is not None:
            self.b_obs[self.counter] = obs
        if action is not None:
            self.b_act[self.counter][:] = 0
            self.b_act[self.counter][action] = 1
        if action_logits is not None:
            self.b_act_logits[self.counter] = action_logits
        if reward is not None:
            self.b_rew[self.counter] = reward
        if terminal is not None:
            self.b_term[self.counter] = terminal
            if terminal:
                self.agent.summary("steps", self.terminal_step_counter)
                self.terminal_step_counter = 0
            self.terminal_step_counter += 1

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

    def policy_loss(self, actions_onehot, discounted_rewards, predicted_logits):
        log_action_prob = tf.keras.backend.log(
            tf.keras.backend.sum(predicted_logits * actions_onehot, axis=1)
        )
        loss = - log_action_prob * discounted_rewards
        loss = tf.keras.backend.mean(loss)

        return loss


class Metrics:

    def __init__(self):
        self.total_loss = tf.keras.metrics.Mean(name="total_loss", dtype=tf.float32)
        self.training_time = tf.keras.metrics.Mean(name="training_time", dtype=tf.float32)


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

    def observe(self, reward, terminal):
        return self.batch.add(
            reward=tf.cast(reward, tf.float32),
            terminal=tf.cast(terminal, tf.bool),
            increment=True
        )

    @tf.function
    def train(self, observations, actions, discounted_rewards):

        with tf.GradientTape() as tape:
            predicted_logits = self.policy(observations)
            policy_loss = self.policy.policy_loss(actions, discounted_rewards, predicted_logits)
            total_loss = policy_loss

        grads = tape.gradient(total_loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        # Update metrics
        self.summary("policy_loss", policy_loss)
        self.summary("total_loss", total_loss)

        return total_loss

    def summary(self, name, data):
        tf.summary.scalar("%s/%s" % (self.name, name), data, self.optimizer.iterations)


class PGAgent(Agent):

    def __init__(self,
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 gamma=0.99,
                 batch_size=1,
                 dtype=tf.float32,
                 tensorboard_enabled=True,
                 tensorboard_path="./tb/",
                 ):

        super(PGAgent, self).__init__(
            obs_space=obs_space,
            action_space=action_space,
            batch_size=batch_size,
            dtype=dtype,
            policy=PGPolicy(
                action_space=action_space,
                dtype=dtype
            ),
            tensorboard_enabled=tensorboard_enabled,
            tensorboard_path=tensorboard_path
        )

        self.gamma = gamma
        self.action_space = action_space

    def reset(self):
        self.batch.counter = 0

    def predict(self, observation):
        action_logits = self.policy.predict(observation)

        action_sample = np.random.choice(np.arange(self.action_space), p=tf.squeeze(action_logits).numpy())

        self.batch.add(obs=observation, action_logits=action_logits, action=action_sample)
        return action_sample

    def observe(self, reward, terminal):
        super().observe(reward, terminal)

        if self.batch.counter == 0:

            s = time.time()
            loss = self.train(
                self.batch.b_obs,
                self.batch.b_act,
                self.G(
                    self.batch.b_rew,
                    self.batch.b_term
                )
            )
            self.metrics.total_loss(loss)
            self.metrics.training_time(time.time() - s)

            logging.debug("Training loss %f | Time Elapsed: %f", self.metrics.total_loss.result(), self.metrics.training_time.result())
            self.summary("training_time", self.metrics.training_time.result())
            self.summary("training_loss_smooth", self.metrics.total_loss.result())

    """
    # G is commonly refered to as the cumulative discounted rewards
    """
    def G(self, rewards, terminals):
        """ take 1D float array of rewards and compute discounted reward """

        discounted_rewards = np.zeros_like(rewards)

        cum_r = 0
        for i in reversed(range(0, len(rewards))):
            cum_r = rewards[i] + (cum_r * self.gamma) * (1 - terminals[i])

            discounted_rewards[i] = cum_r

        discounted_rewards = (discounted_rewards - discounted_rewards.std()) / discounted_rewards.mean()

        return discounted_rewards


if __name__ == "__main__":

    env = gym.make('CartPole-v0')
    agent = PGAgent(
        obs_space=env.observation_space,
        action_space=env.action_space.n,
        batch_size=128,
        tensorboard_enabled=True,
        tensorboard_path="./tb/"
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
            agent.observe(reward, terminal)
            steps += 1