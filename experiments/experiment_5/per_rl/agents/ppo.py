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

from experiments.experiment_5.per_rl.agents.reinforce import REINFORCE


@DecoratedAgent
class PPO(Agent):
    PARAMETERS = ["gae_lambda", "epsilon", "kl_coef", "value_coef", "gamma"]
    DEFAULTS = defaults.PPO

    def __init__(self, **kwargs):
        super(PPO, self).__init__(**kwargs)

        self.add_preprocessor("returns", self.generalized_advantage_estimation)

        self.add_loss("policy_loss", self.clipped_surrogate_loss, "Policy loss of PPO")
        self.add_loss("value_loss", self.action_value_loss, "Action loss of PPO")

    # TODO make a way to adopt this from other classes i.e reinforce.
    def _predict(self, inputs):
        pred = super()._predict(inputs)
        action = tf.squeeze(tf.random.categorical(pred["logits"], 1))
        self.data["action"] = tf.one_hot(action, self.action_space)
        return action.numpy()

    def action_value_loss(self, old_action_value, action_value, returns, **kwargs):
        """
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))"""

        v_pred_clipped = old_action_value + tf.clip_by_value(
            action_value - old_action_value,
            -self.args["epsilon"], self.args["epsilon"]
        )

        vf_losses_1 = tf.square(action_value - returns)

        vf_losses_2 = tf.square(v_pred_clipped - returns)

        vf_loss = self.args["value_coef"] * tf.reduce_mean(tf.maximum(vf_losses_1, vf_losses_2))

        return vf_loss

        # return tf.losses.mean_squared_error(old_action_value, action_value) # what.. this works?
        #return tf.losses.mean_squared_error(returns, action_value) * self.args["value_coef"]



    def clipped_surrogate_loss(self, old_logits, action, old_action_value, advantage, logits, **kwargs):
        neglogp_old = tf.losses.categorical_crossentropy(action, old_logits)
        neglogp_new = tf.losses.categorical_crossentropy(action, logits)

        "Conservative Policy Iteration with multiplied advantages"
        l_cpi = tf.exp(neglogp_old - neglogp_new)

        # Ratio loss with neg advantages
        pg_losses = -advantage * l_cpi

        """Clipping the l_cpi according to paper."""
        pg_losses_clipped = -advantage * tf.clip_by_value(l_cpi, 1.0 - self.args["epsilon"], 1.0 + self.args["epsilon"])

        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses_clipped))

        # Metrics
        approxkl = .5 * tf.reduce_mean(tf.square(neglogp_new - neglogp_old))
        clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(l_cpi - 1.0), self.args["epsilon"]), dtype=tf.float64))

        self.metrics.add("approxkl", approxkl, ["mean"], "train", epoch=True)
        self.metrics.add("clipfrac", clipfrac, ["mean"], "train", epoch=True)

        return -pg_loss

    def discount(self, x, gamma):
        return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def generalized_advantage_estimation(self, policy, obs1, old_action_value, terminal, reward, **kwargs):

        last_gae = 0.0
        result_adv = []
        result_ref = []

        V = old_action_value
        V_1 = np.concatenate((old_action_value[1:], [policy(obs1[-1:])["action_value"]]))

        for i in reversed(range(self.batch.counter)):
            if terminal[i]:
                delta = reward[i] - V[i]
                last_gae = delta
            else:
                delta = reward[i] + self.args["gamma"] * V_1[i] - V[i]
                last_gae = delta + self.args["gamma"] * self.args["gae_lambda"] * last_gae

            result_adv.append(last_gae)
            result_ref.append(last_gae + V[i])

        result_adv = np.asarray(result_adv)
        result_ref = np.asarray(result_ref)

        # Normalize mini-batch
        for i in range(0, self.batch.counter, self.batch.mbsize):
            sub_adv = result_adv[i:i + self.batch.mbsize]
            result_adv[i:i + self.batch.mbsize] = (sub_adv - sub_adv.mean()) / np.maximum(sub_adv.std(), 1e-6)

        # Normalize whole batch.
        #result_adv = (result_adv - result_adv.mean()) / np.maximum(result_adv.std(), 1e-6)

        return dict(
            returns=result_ref,
            advantage=result_adv
        )

        # Predict the v_next (Aka the value from the last observation before batch was complete.
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