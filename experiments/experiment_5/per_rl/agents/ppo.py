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

import tensorflow as tf
from experiments.experiment_5.per_rl.agents.a2c import A2C
from experiments.experiment_5.per_rl.agents.agent import Agent
from experiments.experiment_5.per_rl.agents.configuration import defaults


class PPO(A2C):
    DEFAULTS = defaults.PPO

    def __init__(self, epsilon=0.1, **kwargs):
        super(PPO, self).__init__(**Agent.arguments())
        self.epsilon = epsilon  # Clipping coefficient

        self.add_loss(
            "clipped_surrogate_loss",
            lambda prediction, data: self.clipped_surrogate_loss(
                self.inference_policy(data["obs"])["policy_logits"],
                prediction["policy_logits"],
                data["actions"],
                # self.discounted_returns(data["rewards"], data["terminals"])

                self.advantage(
                    self.inference_policy,
                    data["obs"],
                    data["obs1"],
                    data["rewards"],
                    data["terminals"]
                )

            )
        )

        """Remove standard REINFORCE (pg) loss."""
        self.remove_loss("policy_loss")

    def clipped_surrogate_loss(self, PI_old, PI_new, A, ADV):

        neg_log_old = tf.reduce_sum(-tf.math.log(tf.clip_by_value(PI_old, 1e-7, 1)) * A, axis=1)
        neg_log_new = tf.reduce_sum(-tf.math.log(tf.clip_by_value(PI_new, 1e-7, 1)) * A, axis=1)

        "Conservative Policy Iteration with multiplied advantages"
        l_cpi = tf.exp(neg_log_old - neg_log_new) * ADV

        """Clipping the l_cpi according to paper."""
        l_cpi_clipped = tf.clip_by_value(l_cpi, 1.0 - self.epsilon, 1.0 + self.epsilon) * ADV

        """Use the lowest of l_cpi or clipped"""
        l_clip = tf.minimum(l_cpi, l_cpi_clipped)

        """Calculate the loss."""
        pg_loss = -tf.reduce_mean(l_clip)

        return pg_loss

    def action_value_loss(self, returns, predicted):
        return -super().action_value_loss(returns, predicted)
