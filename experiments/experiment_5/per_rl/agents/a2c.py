import tensorflow as tf

from experiments.experiment_5.per_rl import utils
from experiments.experiment_5.per_rl.agents.agent import Agent
from experiments.experiment_5.per_rl.agents.configuration import defaults
from experiments.experiment_5.per_rl.agents.reinforce import REINFORCE


class A2C(REINFORCE):
    DEFAULTS = defaults.A2C

    def __init__(self,
                 value_coef=0.5,  # For action_value_loss, we multiply by this factor
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


        self.add_operation("advantage", self.advantage)
        self.add_loss("action_value_loss", self.action_value_loss)

    def advantage(self, action_value, returns, obs1, policy, **kwargs):
        #V1 = policy(obs1)["action_value"] # (V1*self.gamma)

        advantage = returns - tf.stop_gradient(action_value)
        self.metrics.add("explained-variance", utils.explained_variance(action_value, returns), ["sum_mean_frequent", "mean_total"], "loss")

        return advantage

    def action_value_loss(self, action_value, advantage, returns, **kwargs):
        """
        The action_value loss is the MSE of discounted reward and predicted
        :param returns:
        :param predicted:
        :return:
        """
        if self.value_loss == "huber":

            loss = tf.losses.Huber()(tf.stop_gradient(returns), action_value)

        elif self.value_loss == "mse":
            loss = tf.losses.mean_squared_error(tf.stop_gradient(returns), action_value)
        else:
            raise NotImplementedError("The loss %s is not implemented for %s." % (self.value_loss, self.name))

        return self.value_coef * loss
