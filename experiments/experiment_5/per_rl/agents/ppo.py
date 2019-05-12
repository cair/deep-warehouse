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
from scipy.signal import lfilter
import tensorflow as tf
from experiments.experiment_5.per_rl.agents.a2c import A2C
from experiments.experiment_5.per_rl.agents.agent import Agent
from experiments.experiment_5.per_rl.agents.configuration import defaults
import numpy as np

class PPO(A2C):
    DEFAULTS = defaults.PPO

    def __init__(self, gae_lambda=0.95, epsilon=0.2, **kwargs):
        super(PPO, self).__init__(**Agent.arguments())
        self.epsilon = epsilon  # Clipping coefficient
        self.gae_lambda = gae_lambda

        self.add_operation("returns", self.generalized_advantage_estimation)
        self.add_operation("old_logits", self.old_logits)

    def old_logits(self, inputs, **kwargs):

        pi_old = self.inference_policy(inputs)
        return pi_old["logits"]

    def policy_loss(self, logits, action, advantage, **kwargs):
        return self.clipped_surrogate_loss(logits, action, advantage, **kwargs)

    def clipped_surrogate_loss(self, logits, action, advantage, old_logits, **kwargs):

        neg_log_old = tf.reduce_sum(-tf.math.log(tf.clip_by_value(old_logits, 1e-7, 1)) * action, axis=1)
        neg_log_new = tf.reduce_sum(-tf.math.log(tf.clip_by_value(logits, 1e-7, 1)) * action, axis=1)

        "Conservative Policy Iteration with multiplied advantages"
        l_cpi = tf.exp(neg_log_old - neg_log_new)

        # Ratio loss with neg advantages
        pg_losses = -advantage * l_cpi

        """Clipping the l_cpi according to paper."""
        pg_losses_clipped = tf.clip_by_value(l_cpi, 1.0 - self.epsilon, 1.0 + self.epsilon) * -advantage

        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses_clipped))

        return pg_loss

    def discount(self, x, gamma):
        return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def generalized_advantage_estimation(self, policy, obs1, action_value, terminal, reward, **kwargs):
        # TODO - Use action_value or append obs1 vpred?

        action_value_new = np.concatenate((action_value[1:], [policy(obs1[-1:])["action_value"]]))
        action_value_old = action_value
        terminal_new = np.concatenate((terminal[1:], [0]))

        advantage = self.discount(reward + self.gamma * action_value_new * (1 - terminal_new) - action_value_old, self.gamma * self.gae_lambda)
        td_lambda_return = advantage + action_value_old
        advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)

        return dict(
            returns=np.asarray(td_lambda_return),
            advantage=np.asarray(advantage)
        )