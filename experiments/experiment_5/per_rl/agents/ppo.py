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
from scipy.signal import lfilter
import tensorflow as tf

from experiments.experiment_5.per_rl import utils
from experiments.experiment_5.per_rl.agents.a2c import A2C
from experiments.experiment_5.per_rl.agents.agent import Agent, DecoratedAgent
from experiments.experiment_5.per_rl.agents.configuration import defaults
import numpy as np

# https://learningai.io/projects/2017/07/28/ai-gym-workout.html

# PRUNING - NOT DONE IN LITERATURE!
# https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a?fbclid=IwAR0szRFNxYr7c7o9-UK0t8hhs7Tr4pcjjs1bvUsksQvyJLH61U7-7n7tBFs

# https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-spring18/slides/cs885-lecture15b.pdf
from experiments.experiment_5.per_rl.agents.reinforce import REINFORCE


@DecoratedAgent
class PPO(Agent):
    PARAMETERS = [
        "gae_lambda",
        "gae",
        "epsilon",
        "kl_coef",
        "vf_coeff",
        "vf_clipping",
        "vf_clip_param",
        "gamma",
        "entropy_coef"
    ]
    DEFAULTS = defaults.PPO

    def __init__(self, **kwargs):
        super(PPO, self).__init__(**kwargs)

        self.add_batch_preprocessor("advantages", self.advantages)
        self.add_mb_preprocessor("advantages", self.mb_advantages)

        self.add_loss("policy_loss", self.policy_loss, "Policy loss of PPO")
        self.add_loss("value_loss", self.value_loss, "Action loss of PPO")
        self.add_loss("entropy_loss", self.entropy_loss, "Action loss of PPO")
        # TODO make a way to adopt this from other classes i.e reinforce.

    def _predict(self, inputs):
        pred = super()._predict(inputs)
        action = tf.squeeze(tf.random.categorical(pred["logits"], 1))
        self.data["action"] = tf.one_hot(action, self.action_space)
        return action.numpy()

    def value_loss(self, old_values, values, returns, **kwargs):

        if self.args["vf_clipping"]:
            v_pred_clipped = \
                old_values + tf.clip_by_value(
                    values - old_values,
                    -self.args["vf_clip_param"],
                    self.args["vf_clip_param"]
                )
            vf_clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(v_pred_clipped - 1.0), self.args["vf_clip_param"]), dtype=tf.float64))
            self.metrics.add("vf_clipfrac", vf_clipfrac, ["mean"], "train", epoch=True)
            vf_losses_1 = tf.square(values - returns)
            vf_losses_2 = tf.square(v_pred_clipped - returns)
            vf_loss = tf.reduce_mean(tf.maximum(vf_losses_1, vf_losses_2))

            return self.args["vf_coeff"] * vf_loss
        else:
            return tf.losses.mean_squared_error(returns, values) * self.args["vf_coeff"]


    def policy_loss(self, logits, old_logits, action, advantages, **kwargs):

        neglogpac_old = tf.reduce_sum(tf.math.log_softmax(old_logits, axis=1) * action, axis=1)
        neglogpac_new = tf.reduce_sum(tf.math.log_softmax(logits, axis=1) * action, axis=1)

        ratios = tf.exp(neglogpac_new - tf.stop_gradient(neglogpac_old))

        surr1 = ratios * advantages
        surr2 = tf.clip_by_value(ratios, 1.0 - self.args["epsilon"], 1.0 + self.args["epsilon"])

        # Metrics
        #approxkl = .5 * tf.reduce_mean(tf.square(neglogpac_new - neglogpac_old))
        #self.metrics.add("approxkl", approxkl, ["mean"], "train", epoch=True)

        clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratios - 1.0), self.args["epsilon"]), dtype=tf.float64))
        self.metrics.add("policy_clipfrac", clipfrac, ["mean"], "train", epoch=True)

        return -tf.reduce_mean(tf.minimum(surr1, surr2))

    def entropy_loss(self, logits, **kwargs):

        log_prob = tf.math.softmax(logits)
        entropy = self.args["entropy_coef"] * tf.reduce_mean(tf.reduce_sum(log_prob * tf.math.log(log_prob), axis=1))
        return -entropy

    def discount(self, x, gamma):
        return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


    def mb_advantages(self, returns, values, old_values, **kwargs):

        self.metrics.add(
            "explained-variance",
            utils.explained_variance(old_values, returns), ["mean"], "loss", epoch=True, total=True)

        # Returns = R + yV(s')
        advantage = np.asarray(returns - values)
        # Normalize the advantages
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        return advantage

    def generalized_advantage_estimation(self, old_values, last_obs, policy, rewards, terminals, **kwargs):
        V = np.concatenate((old_values, [policy([last_obs])["values"]]))
        terminal = np.concatenate((terminals, [0]))
        gamma = self.args["gamma"]
        lam = self.args["gae_lambda"]
        adv = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(self.batch.counter)):
            nextnonterminal = 1 - terminal[t+1]
            delta = rewards[t] + gamma * (V[t+1] * nextnonterminal) - V[t]
            adv[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam

        returns = adv + old_values

        return dict(
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
            returns=discounted_rewards
        )

    def advantages(self, **kwargs):
        if self.args["gae"]:
            return self.generalized_advantage_estimation(**kwargs)
        else:
            return self.discounted_returns(**kwargs)

