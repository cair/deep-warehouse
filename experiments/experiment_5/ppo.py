from experiments.experiment_5.a2c import A2C
from experiments.experiment_5.agent import Agent
from experiments.experiment_5.network import PGPolicy, Policy
from experiments.experiment_5.pg import REINFORCE
import tensorflow as tf


class PPOPolicy(Policy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.h_1 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.h_2 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.h_3 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.h_4 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)

        self.logits = tf.keras.layers.Dense(self.agent.action_space, activation="softmax", name='policy_logits', dtype=self.agent.dtype)

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
        policies=dict(
            target=lambda agent: PPOPolicy(
                agent=agent,
                inference=True,
                training=True,
                optimizer=tf.keras.optimizers.Adam(lr=0.001)
            ),
            old=lambda agent: PPOPolicy(
                agent=agent,
                inference=False,
                training=False,
                optimizer=tf.keras.optimizers.Adam(lr=0.001)
            ),
        )
    )

    # TODO
    # * Add KL-penalized objective (loss)
    # * Clipped surrogate objective
    # * generalized advantage estimation  - WHEN RNN
    # * finite horizon estimators
    # * entropy bonus
    def __init__(self,
                 value_coef=0.5,  # For action_value_loss, we multiply by this factor
                 value_loss="huber",
                 clipping_threshold=0.2,

                 entropy_coef=0.0001,
                 **kwargs):
        super(PPO, self).__init__(**Agent.arguments())

        self.add_loss(
            "clipped_surrogate_loss",
            lambda pred: self.clipped_surrogate_loss(
                self.batch.obs(),
                pred["policy_logits"]
            )
        )

    def clipped_surrogate_loss(self, obs, policy):


        r = policy / self.policies["old"](obs)["policy_logits"]
        print(r.shape)
        return tf.cast(1, dtype=tf.float32)
        # r_t(theta)
        #pred_old = self.policies["old"](state)
