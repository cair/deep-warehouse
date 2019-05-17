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

        self.add_preprocessor("advantage", self.advantage)

        self.add_loss("policy_loss", self.policy_loss, "Policy loss of PPO")
        self.add_loss("value_loss", self.value_loss, "Action loss of PPO")
        self.add_loss("entropy_loss", self.entropy_loss, "Action loss of PPO")
    # TODO make a way to adopt this from other classes i.e reinforce.

    def _predict(self, inputs):
        pred = super()._predict(inputs)
        action = tf.squeeze(tf.random.categorical(pred["logits"], 1))
        self.data["action"] = tf.one_hot(action, self.action_space)
        return action.numpy()

    def value_loss(self, old_action_value, action_value, returns, **kwargs):

        if self.args["vf_clipping"]:
            v_pred_clipped = \
                old_action_value + tf.clip_by_value(
                    action_value - old_action_value,
                    -self.args["vf_clip_param"],
                    self.args["vf_clip_param"]
                )

            vf_losses_1 = tf.square(action_value - returns)
            vf_losses_2 = tf.square(v_pred_clipped - returns)
            vf_loss = tf.reduce_mean(tf.maximum(vf_losses_1, vf_losses_2))

            return self.args["vf_coeff"] * vf_loss
        else:
            return tf.losses.mean_squared_error(returns, action_value) * self.args["vf_coeff"]

    def policy_loss(self, logits, old_logits, action, advantage, **kwargs):

        neglogpac_old = tf.reduce_sum(tf.math.log_softmax(old_logits, axis=1) * action, axis=1)
        neglogpac_new = tf.reduce_sum(tf.math.log_softmax(logits, axis=1) * action, axis=1)

        ratios = tf.exp(neglogpac_new - tf.stop_gradient(neglogpac_old))

        surr1 = ratios * advantage
        surr2 = tf.clip_by_value(ratios, 1.0 - self.args["epsilon"], 1.0 + self.args["epsilon"])

        # Metrics
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac_new - neglogpac_old))
        clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratios - 1.0), self.args["epsilon"]), dtype=tf.float64))

        self.metrics.add("approxkl", approxkl, ["mean"], "train", epoch=True)
        self.metrics.add("clipfrac", clipfrac, ["mean"], "train", epoch=True)


        return -tf.reduce_mean(tf.minimum(surr1, surr2))

    def entropy_loss(self, logits, **kwargs):
        #entropy_loss = -tf.reduce_mean(tf.losses.categorical_crossentropy(logits, logits, from_logits=True) * self.entropy_coef)

        log_prob = tf.math.softmax(logits)
        entropy = self.args["entropy_coef"] * tf.reduce_mean(tf.reduce_sum(log_prob * tf.math.log(log_prob), axis=1))
        return -entropy

    def discount(self, x, gamma):
        return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def generalized_advantage_estimation(self, old_action_value, policy, obs1, reward, terminal, **kwargs):
        """V_1 = np.concatenate((old_action_value[1:], [policy(obs1[-1:])["action_value"]]))
        V = old_action_value
        T_1 = np.concatenate((terminal[1:], [0]))

        advantage = self.discount(
            reward +
            self.args["gamma"] * V_1 * (1 - T_1) -
            V, self.args["gamma"] * self.args["gae_lambda"]
        )

        td_lambda_return = advantage + V
        advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)

        return dict(
            returns=np.asarray(td_lambda_return),
            advantage=np.asarray(advantage)
        )"""

        last_gae = 0
        adv = np.zeros_like(reward)
        ret = np.zeros_like(reward)

        V = old_action_value
        V_1 = np.concatenate((old_action_value[1:], [policy(obs1[-1:])["action_value"]]))

        for i in reversed(range(0, self.batch.counter)):
            if terminal[i]:
                delta = reward[i] - V[i]
                last_gae = delta
            else:
                delta = reward[i] + self.args["gamma"] * V_1[i] - V[i]
                last_gae = delta + self.args["gamma"] * self.args["gae_lambda"] * last_gae

            adv[i] = last_gae
            ret[i] = last_gae + V[i]

        adv = (adv - adv.std()) / adv.mean()

        return dict(
            advantage=adv,
            returns=ret
        )

    def discounted_returns(self, reward, terminal, old_action_value, **kwargs):
        discounted_rewards = np.zeros_like(reward)
        advantage = np.zeros_like(reward)
        cum_r = 0
        for i in reversed(range(0, self.batch.counter)):
            cum_r = reward[i] + cum_r * self.args["gamma"] * (1 - terminal[i])

            discounted_rewards[i] = cum_r
            advantage[i] = cum_r - old_action_value[i]

        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)

        return dict(
            advantage=advantage,
            returns=discounted_rewards
        )

    def advantage(self, **kwargs):
        if self.args["gae"]:
            return self.generalized_advantage_estimation(**kwargs)
        else:
            return self.discounted_returns(**kwargs)

