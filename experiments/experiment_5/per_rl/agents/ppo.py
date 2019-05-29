# TODO
# * Add KL-penalized objective (loss)
# * Clipped surrogate objective
# * generalized advantage estimation  - WHEN RNN
# * finite horizon estimators
# * entropy bonus
# L2 regularization on networks. (Needed?)
# https://nervanasystems.github.io/coach/components/agents/policy_optimization/cppo.html
# https://medium.com/mlreview/making-sense-of-the-bias-variance-trade-off-in-deep-reinforcement-learning-79cf1e83d565 Variance
# https://github.com/hill-a/stable-baselines
# Much info here about stop gradient https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/rl/ppo.py also how to use Distributions.
# https://github.com/tensorflow/agents
# TODO
# https://github.com/Anjum48/rl-examples/blob/master/ppo/ppo_joined.py very nice implementation using dataset...
# https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe

# DO SOME https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/ppo/ppo.py
import time

import tensorflow as tf

from experiments.experiment_5.per_rl import utils
from experiments.experiment_5.per_rl.agents.agent import Agent, DecoratedAgent
from experiments.experiment_5.per_rl.agents.configuration import defaults
import numpy as np

# https://learningai.io/projects/2017/07/28/ai-gym-workout.html

# PRUNING - NOT DONE IN LITERATURE!
# https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a?fbclid=IwAR0szRFNxYr7c7o9-UK0t8hhs7Tr4pcjjs1bvUsksQvyJLH61U7-7n7tBFs

# https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-spring18/slides/cs885-lecture15b.pdf
from experiments.experiment_5.per_rl.agents.reinforce import REINFORCE

# WEIGHT AVERAGING
# http://www.gatsby.ucl.ac.uk/~balaji/udl-camera-ready/UDL-24.pdf


# PPO-CMA
# https://www.reddit.com/r/reinforcementlearning/comments/boi4m0/ppocma_proximal_policy_optimization_with/
from experiments.experiment_5.per_rl.distribution.categorical import Categorical
from experiments.experiment_5.per_rl.utils.decorators import KeepLocals


@DecoratedAgent
class PPO(Agent):
    PARAMETERS = [
        "normalize_advantages",
        "gae_lambda",
        "gae",
        "epsilon",
        "kl_coeff",
        "vf_coeff",
        "vf_clipping",
        "vf_clip_param",
        "gamma",
        "entropy_coef"
    ]
    DEFAULTS = defaults.PPO

    def __init__(self, **kwargs):
        super(PPO, self).__init__(**kwargs)

        self.add_processor("advantages", self.advantages, "batch")
        self.add_processor("neglogp", self.mb_neglogp, "mini-batch")
        self.add_processor("update_kl", self.post_update_kl, "post")

        #self.add_processor("action_kl_loss", self.kl_loss, "loss", text="Action KL Loss")
        self.add_processor("policy_loss", self.policy_loss, "loss", text="Policy loss of PPO")
        self.add_processor("value_loss", self.value_loss, "loss", text="Action loss of PPO")
        self.add_processor("entropy_loss", self.entropy_loss, "loss", text="Action loss of PPO")
        self.add_callback(self.on_terminal, "on_terminal")

    def on_terminal(self):
        self.metrics.add("predicted_value", self.data["old_values"], ["mean"], "reward", episode=True, epoch=False,
                         total=False)

    def _predict(self, inputs):
        pred = super()._predict(inputs)
        logits = pred["logits"]

        sampled = tf.squeeze(tf.random.categorical(logits, 1))
        return sampled.numpy()

    def value_loss(self, old_values, values, returns, **kwargs):

        if self.args["vf_clipping"]:
            v_pred_clipped = \
                old_values + tf.clip_by_value(
                    values - old_values,
                    -self.args["vf_clip_param"],
                    self.args["vf_clip_param"]
                )
            vf_clipfrac = tf.reduce_mean(
                tf.cast(tf.greater(tf.abs(v_pred_clipped - 1.0), self.args["vf_clip_param"]), dtype=tf.float64))
            self.metrics.add("vf_clipfrac", vf_clipfrac, ["mean"], "train", epoch=True)
            vf_losses_1 = tf.square(values - returns)
            vf_losses_2 = tf.square(v_pred_clipped - returns)
            vf_loss = tf.reduce_mean(tf.maximum(vf_losses_1, vf_losses_2))

        else:

            vf_loss = tf.reduce_mean(tf.math.squared_difference(returns, tf.reduce_sum(values, axis=0)))
            # vf_loss = tf.losses.mean_squared_error(returns, values)

        return vf_loss * self.args["vf_coeff"]

    # @KeepLocals(
    #    include=["kl"],
    #    attribute="udata"
    # )
    def kl_loss(self, old_logits, logits, actions, **kwargs):
        # action_kl = tf.losses.kullback_leibler_divergence(neglogp_old, neglogp_new)
        # print(action_kl)
        # Metrics
        # approxkl = .5 * tf.reduce_mean(tf.square(neglogpac_new - neglogpac_old))
        # self.metrics.add("approxkl", approxkl, ["mean"], "train", epoch=True)

        logp = tf.math.softmax(logits, axis=1) * actions
        logq = tf.math.softmax(old_logits, axis=1) * actions

        kl = tf.reduce_mean(tf.reduce_sum(tf.math.softmax(logits) * (logp - logq), axis=1))

        # action_kl = tf.reduce_mean(tf.square(neglogp_new - neglogp_old))  # APPROX VERSION?

        # print(action_kl)
        return kl * self.args["kl_coeff"]

    def policy_loss(self, neglogpac_old, neglogpac_new, advantages, **kwargs):

        surr1 = tf.exp(tf.stop_gradient(neglogpac_old) - neglogpac_new)
        surr2 = tf.clip_by_value(
            surr1,
            1.0 - self.args["epsilon"],
            1.0 + self.args["epsilon"]
        )

        surr_clip_frac = tf.reduce_mean(
            tf.cast(tf.greater(tf.abs(surr1 - 1.0), self.args["epsilon"]), dtype=tf.float64)
        )

        self.metrics.add("policy_clipfrac", surr_clip_frac, ["mean"], "train", epoch=True)

        return -tf.reduce_mean(tf.minimum(
            surr1 * advantages,
            surr2 * advantages
        ))

    def entropy_loss(self, logits, **kwargs):
        return self.args["entropy_coef"] * -tf.reduce_mean(Categorical.entropy(logits))

    def mb_neglogp(self, old_logits, logits, actions, **kwargs):

        return dict(
            neglogpac_old=Categorical.neglogpac(old_logits, actions),
            neglogpac_new=Categorical.neglogpac(logits, actions)
        )

    def generalized_advantage_estimation(self, old_values, last_obs, policy, rewards, terminals, **kwargs):

        V = np.concatenate((old_values, [policy([last_obs])["values"]]))
        terminal = np.concatenate((terminals, [0]))
        gamma = self.args["gamma"]
        lam = self.args["gae_lambda"]
        adv = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(self.batch.counter)):
            nextnonterminal = 1 - terminal[t + 1]
            delta = rewards[t] + gamma * (V[t + 1] * nextnonterminal) - V[t]
            adv[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam

        returns = adv + old_values

        return dict(
            advantages=adv,
            returns=returns
        )

    def discounted_returns(self, rewards, terminals, old_values, **kwargs):
        discounted_rewards = np.zeros_like(rewards)
        advantage = np.zeros_like(rewards)
        cum_r = 0
        for i in reversed(range(0, self.batch.counter)):
            cum_r = rewards[i] + cum_r * self.args["gamma"] * (1 - terminals[i])

            discounted_rewards[i] = cum_r
            advantage[i] = cum_r + old_values[i]

        return dict(
            advantages=advantage,
            returns=discounted_rewards
        )

    def advantages(self, **kwargs):

        data = self.generalized_advantage_estimation(**kwargs) if self.args["gae"] else \
            self.discounted_returns(**kwargs)

        if self.args["normalize_advantages"]:
            data["advantages"] = (data["advantages"] - data["advantages"].mean()) / (data["advantages"].std() + 1e-8)

        return data


    def post_update_kl(self, **kwargs):
        return {

        }
