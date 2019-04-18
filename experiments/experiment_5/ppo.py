from experiments.experiment_5.a2c import A2C
from experiments.experiment_5.agent import Agent
from experiments.experiment_5.network import PGPolicy, Policy
from experiments.experiment_5.pg import REINFORCE
import tensorflow as tf
import tensorflow_probability as tfp

class PPOPolicy(Policy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.h_1 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.h_2 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.h_3 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.h_4 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)

        self.logits = tf.keras.layers.Dense(self.agent.action_space, activation="softmax", name='policy_logits',
                                            dtype=self.agent.dtype)

        self.state_value_1 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.state_value = tf.keras.layers.Dense(1,
                                                 activation="linear",
                                                 name="state_value",
                                                 dtype=self.agent.dtype
                                                 )

    def call(self, inputs):
        x = self.h_1(inputs)
        x = self.h_2(x)
        x = self.h_3(x)
        x = self.h_4(x)

        policy_logits = self.logits(x)
        state_value = self.state_value(self.state_value_1(x))

        return {
            "policy_logits": policy_logits,
            "action_value": state_value
        }


class PPO(A2C):
    DEFAULTS = dict(
        batch_mode="steps",
        batch_size=64,
        entropy_coef=0.01,  # Entropy should be 0.0 for continous action spaces.  # TODO
        value_coef=1.0,
        value_loss="huber",
        max_grad_norm=None,
        baseline="reward_mean",
        policies=dict(
            # The training policy (The new one)
            target=lambda agent: PPOPolicy(
                agent=agent,
                inference=False,
                training=True,
                optimizer=tf.keras.optimizers.RMSprop(lr=0.001)
            ),
            # The old policy (The inference one)
            old=lambda agent: PPOPolicy(
                agent=agent,
                inference=True,
                training=False,
                optimizer=tf.keras.optimizers.Adam(lr=0.001)
            ),
        ),
        policy_update=dict(
            interval=5,  # Update every 10 training epochs,
            strategy="copy"
        )
    )

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

    # TODO
    # https://github.com/Anjum48/rl-examples/blob/master/ppo/ppo_joined.py very nice implementation using dataset...
    def __init__(self,
                 epsilon=0.1,
                 **kwargs):
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
