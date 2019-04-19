from experiments.experiment_5 import utils
from experiments.experiment_5.agent import Agent
from experiments.experiment_5.network import PGPolicy
from experiments.experiment_5.reinforce import REINFORCE
import tensorflow as tf
import numpy as np

class A2CPolicy(PGPolicy):
    """
    Nice resources:
    Blog: http://steven-anker.nl/blog/?p=184
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.h_4 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.action_value = tf.keras.layers.Dense(1, dtype=self.agent.dtype)

    def call(self, inputs):
        data = super().call(inputs)

        x = self.shared(inputs)
        x = self.h_4(x)
        action_value = self.action_value(x)

        data["action_value"] = action_value
        return data


class A2C(REINFORCE):
    DEFAULTS = dict(
        batch_mode="steps",
        batch_size=64,
        entropy_coef=0.0001,
        policies=dict(
            policy=lambda agent: A2CPolicy(
                agent=agent,
                inference=True,
                training=False,
                optimizer=None
            ),
            target=lambda agent: A2CPolicy(
                agent=agent,
                inference=False,
                training=True,
                optimizer=tf.keras.optimizers.Adam(lr=0.001)  # decay=0.99, epsilon=1e-5)

            )
        ),
        policy_update=dict(
            interval=5,  # Update every 5 training epochs,
            strategy="copy",  # "copy, mean"
        )
    )

    def __init__(self,
                 value_coef=1,  # For action_value_loss, we multiply by this factor
                 value_loss="mse",
                 tau=0.95,
                 **kwargs):
        super(A2C, self).__init__(**Agent.arguments())

        self.value_coef = value_coef
        self.value_loss = value_loss
        self.tau = tau

        self.metrics.text("explained_variance", "Explained Variance is an attempt to measure the quality of the state "
                                                "value.  \n"
                                                "**ev=0**: Might as well have predicted zero  \n"
                                                "**ev=1**: Perfect prediction  \n  "
                                                "**ev<0**: Worse than just predicting zero")

        # self.add_calculation("advantage", self.advantage)

        # self.add_loss("policy_loss", self.policy_loss)  # Overrides loss of REINFORCE
        self.add_loss("action_value_loss", self.action_value_loss)

    def G(self, data, **kwargs):
        """Override G of REINFORCE"""
        super().G(data, **kwargs)
        discounted_rewards = data["G"]

        action_values = data["action_value"]
        next_values = tf.squeeze(tf.concat([action_values[1:], data["policy"](data["obs1"])["action_value"]], axis=0))
        action_values = tf.squeeze(action_values)

        advantage = discounted_rewards + (self.gamma*next_values - action_values)
        # TODO think this is correct. must check with other implementations if next_values is correct....
        self.metrics.add("explained_variance", utils.explained_variance(action_values, discounted_rewards))

        data["advantage"] = advantage


    def advantage(self, policy, obs, obs1, rewards, terminals):
        R = self.discounted_returns(rewards, terminals)
        values = tf.squeeze(policy(obs)["action_value"])
        next_value = tf.squeeze(policy(obs1)["action_value"])

        return R + ((next_value * self.gamma) - values)

    def action_value_loss(self, action_value=None, advantage=None, **kwargs):
        """
        The action_value loss is the MSE of discounted reward and predicted
        :param returns:
        :param predicted:
        :return:
        """
        """if self.value_loss == "huber":
            loss = tf.losses.Huber()
            loss = loss(advantage, action_value)
        elif self.value_loss == "mse":
            loss = tf.keras.losses.mean_squared_error(advantage, action_value)
        else:
            raise NotImplementedError("The loss %s is not implemented for %s." % (self.value_loss, self.name))
        """
        loss = tf.reduce_mean(tf.square(advantage))
        #tf.stop_gradient(advantage)

        return self.value_coef * loss
