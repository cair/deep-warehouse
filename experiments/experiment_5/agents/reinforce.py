import tensorflow as tf
import numpy as np
from absl import flags, logging
import time

from experiments.experiment_5.agents.agent import Agent
from experiments.experiment_5.agents.configuration import defaults

FLAGS = flags
logging.set_verbosity(logging.DEBUG)


class REINFORCE(Agent):
    DEFAULTS = defaults.REINFORCE

    def __init__(self,
                 gamma=0.99,
                 entropy_coef=0.001,
                 baseline=None,
                 **kwargs):

        super(REINFORCE, self).__init__(**Agent.arguments())

        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.baseline = baseline

        self.add_operation("returns", self.discounted_returns)
        self.add_operation("advantage", lambda returns, **kwargs: returns)

        self.add_loss("policy_loss", self.policy_loss, "**Policy loss (REINFORCE)**  \nThes policy loss will oscillate with training.")

        if entropy_coef != 0:
            self.add_loss("entropy_loss", self.entropy_loss)

    def predict(self, inputs):
        start = time.perf_counter()
        pred = super().predict(inputs)

        action = tf.squeeze(tf.random.categorical(pred["logits"], 1))

        self.data["action"] = tf.one_hot(action, self.action_space)

        self.metrics.add("inference_time", time.perf_counter() - start, ["mean_total"], "time")
        return action.numpy()

    def observe(self, **kwargs):
        ready = super().observe(**kwargs)

        if ready:

            s = time.perf_counter()

            for batch in self.batch.flush():
                self.train(**batch)

            self.metrics.add("training_time", time.perf_counter() - s, ["mean_total"], "time")

    def policy_loss(self, pred, action=None, advantage=None, **kwargs):
        logits = pred["logits"]
        #neg_log_p = tf.maximum(1e-6, tf.reduce_sum(-tf.math.log(logits) * action, axis=1))
        #return tf.reduce_mean(neg_log_p * advantage)

        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(action, logits=logits)
        loss = tf.reduce_mean(advantage * neg_log_prob)

        return loss

    def entropy_loss(self, pred, obs=None, **kwargs):
        logits = pred["logits"]
        entropy_loss = -tf.reduce_mean(tf.losses.categorical_crossentropy(logits, logits, from_logits=True) * self.entropy_coef)
        return entropy_loss

    def discounted_returns(self, reward, terminal, **kwargs):
        # TODO - Checkout https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py and compare performance

        discounted_rewards = np.zeros_like(reward)

        cum_r = 0
        l = len(reward)
        for i in reversed(range(0, l)):
            cum_r = reward[i] + (cum_r * self.gamma * (1 - terminal[i]))

            discounted_rewards[i] = cum_r

        discounted_rewards = (discounted_rewards - discounted_rewards.std()) / discounted_rewards.mean()
        np.nan_to_num(discounted_rewards, copy=False)

        """Baseline implementations."""
        if self.baseline == "reward_mean":
            baseline = np.mean(reward)
        else:
            baseline = 0

        return discounted_rewards - baseline

    def generalized_advantage_estimation(self, values, next_value, rewards, terminals, gamma=0.99, tau=0.95):
        # TODO not really working ...
        nvalues = np.concatenate((values.numpy(), [next_value]))
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):

            delta = rewards[step] + gamma * nvalues[step + 1] * terminals[step] - nvalues[step]
            gae = delta + gamma * tau * terminals[step] * gae
            returns.insert(0, gae + nvalues[step])

        return returns
