from experiments.experiment_5 import defaults
from experiments.experiment_5.agent import Agent
import tensorflow as tf
import numpy as np
from absl import flags, logging
import time

FLAGS = flags
logging.set_verbosity(logging.DEBUG)


class REINFORCE(Agent):
    DEFAULTS = defaults.REINFORCE


    def __init__(self,
                 gamma=0.99,
                 entropy_coef=0.01,
                 baseline=None,
                 **kwargs):

        super(REINFORCE, self).__init__(**Agent.arguments())

        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.baseline = baseline

        self.add_operation("advantage", self.discounted_returns)
        self.add_loss("policy_loss", self.policy_loss, "**Policy loss (REINFORCE)**  \nThes policy loss will oscillate with training.")

        if entropy_coef != 0:
            self.add_loss("entropy_loss", self.entropy_loss)

    def predict(self, inputs):
        start = time.perf_counter()
        pred = super().predict(inputs)

        action = tf.squeeze(pred["logits"].sample())
        self.data["action"] = tf.one_hot(action, self.action_space)

        self.metrics.add("inference_time", time.perf_counter() - start)
        return action.numpy()

    def observe(self, **kwargs):
        ready = super().observe(**kwargs)

        if ready:
            s = time.perf_counter()


            data = self.batch.flush()
            for batch in data:

                self.train(**batch)

            #for mb in tf.data.Dataset.from_tensor_slices(data):
            #    self.train(**mb)

            self.metrics.add("backprop_time", time.perf_counter() - s, "EpisodicMean")

    """
    # G is commonly refered to as the cumulative discounted rewards
    """



    def policy_loss(self, pred, action=None, advantage=None, **kwargs):
        logits = pred["logits"]
        #neg_log_p = tf.maximum(1e-6, tf.reduce_sum(-tf.math.log(logits) * action, axis=1))
        #return tf.reduce_mean(neg_log_p * advantage)

        neg_log_policy = -logits.log_prob(tf.argmax(action, axis=1))

        return tf.reduce_mean(advantage * neg_log_policy)

    def entropy_loss(self, pred, obs=None, **kwargs):
        logits = pred["logits"]
        entropy_loss = logits.entropy()
        return tf.reduce_mean(-entropy_loss * self.entropy_coef)


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
