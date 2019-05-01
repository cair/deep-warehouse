import tensorflow as tf

from experiments.experiment_5 import utils
from experiments.experiment_5.agents.agent import Agent
from experiments.experiment_5.agents.configuration import defaults
from experiments.experiment_5.agents.reinforce import REINFORCE


class A2C(REINFORCE):
    DEFAULTS = defaults.A2C

    def __init__(self,
                 value_coef=0.5,  # For action_value_loss, we multiply by this factor
                 value_loss="huber",
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

        action_values = data["values"]

        V1 = data["policy"](data["obs1"])["action_value"]
        #advantage = discounted_rewards + (self.gamma*V1 - action_values)

        advantage = discounted_rewards + (V1*self.gamma) - action_values

        self.metrics.add("explained_variance", utils.explained_variance_2d(action_values, discounted_rewards), "EpisodicMean")

        data["returns"] = discounted_rewards
        data["G"] = advantage



    def action_value_loss(self, action_value=None, advantage=None, returns=None, **kwargs):
        """
        The action_value loss is the MSE of discounted reward and predicted
        :param returns:
        :param predicted:
        :return:
        """
        # Must be same shape (squeeze)
        action_value = tf.squeeze(action_value)  # TODO optimize away

        if self.value_loss == "huber":
            loss = tf.keras.losses.Huber()(returns, action_value)

        elif self.value_loss == "mse":
            loss = tf.keras.losses.mean_squared_error(returns, action_value)
        else:
            raise NotImplementedError("The loss %s is not implemented for %s." % (self.value_loss, self.name))

        #loss = tf.reduce_mean(tf.square(action_value - G))
        tf.stop_gradient(returns)

        return self.value_coef * loss
