import tensorflow as tf
import numpy as np
from absl import flags, logging
import time

from experiments.experiment_5.per_rl.agents.agent import Agent, DecoratedAgent
from experiments.experiment_5.per_rl.agents.configuration import defaults

FLAGS = flags
logging.set_verbosity(logging.DEBUG)

@DecoratedAgent
class REINFORCE(Agent):
    PARAMETERS = ["entropy_coef", "gamma", "baseline"]
    DEFAULTS = defaults.REINFORCE

    def __init__(self, **kwargs):
        super(REINFORCE, self).__init__(**kwargs)

        self.add_operation("returns", self.discounted_returns)
        self.add_operation("advantage", lambda returns, **kwargs: returns)

        self.add_loss("policy_loss", self.policy_loss, "**Policy loss (REINFORCE)**  \nThes policy loss will oscillate with training.")

        if self.args["entropy_coef"] != 0:
            self.add_loss("entropy_loss", self.entropy_loss)

    def _predict(self, inputs):
        pred = super()._predict(inputs)
        action = tf.squeeze(tf.random.categorical(pred["logits"], 1))
        self.data["action"] = tf.one_hot(action, self.action_space)

        return action.numpy()

    def policy_loss(self, logits, action, advantage, **kwargs):
        assert logits.shape == action.shape

        log_prob = tf.math.log_softmax(logits, axis=1)
        log_prob = advantage * tf.reduce_sum(log_prob * action, axis=1)

        return -tf.reduce_mean(log_prob)

    def entropy_loss(self, logits, **kwargs):
        #entropy_loss = -tf.reduce_mean(tf.losses.categorical_crossentropy(logits, logits, from_logits=True) * self.entropy_coef)

        log_prob = tf.math.softmax(logits)
        entropy = self.args["entropy_coef"] * tf.reduce_mean(tf.reduce_sum(log_prob * tf.math.log(log_prob), axis=1))
        return -entropy

    def discounted_returns(self, reward, terminal, **kwargs):
        # TODO - Checkout https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py and compare performance

        discounted_rewards = np.zeros_like(reward)

        cum_r = 0
        l = len(reward)
        for i in reversed(range(0, l)):
            cum_r = reward[i] + (cum_r * self.args["gamma"] * (1 - terminal[i]))

            discounted_rewards[i] = cum_r

        discounted_rewards = (discounted_rewards - discounted_rewards.std()) / discounted_rewards.mean()
        np.nan_to_num(discounted_rewards, copy=False)

        """Baseline implementations."""
        if self.args["baseline"] == "reward_mean":
            baseline = np.mean(reward)
        else:
            baseline = 0

        returns = discounted_rewards - baseline
        return returns
