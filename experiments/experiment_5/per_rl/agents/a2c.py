import tensorflow as tf

from experiments.experiment_5.per_rl import utils
from experiments.experiment_5.per_rl.agents.agent import Agent, DecoratedAgent
from experiments.experiment_5.per_rl.agents.configuration import defaults
from experiments.experiment_5.per_rl.agents.reinforce import REINFORCE


@DecoratedAgent
class A2C(REINFORCE):
    PARAMETERS = ["value_coef", "value_loss", "tau"]
    DEFAULTS = defaults.A2C

    def __init__(self, **kwargs):
        super(A2C, self).__init__(**kwargs)

        self.metrics.text("explained_variance", "Explained Variance is an attempt to measure the quality of the state "
                                                "value.  \n"
                                                "**ev=0**: Might as well have predicted zero  \n"
                                                "**ev=1**: Perfect prediction  \n  "
                                                "**ev<0**: Worse than just predicting zero")

        self.add_operation("advantage", self.advantage)
        self.add_loss("action_value_loss", self.action_value_loss)

    def advantage(self, old_action_value, returns, obs1, policy, **kwargs):
        #V1 = policy(obs1)["action_value"] # (V1*self.gamma)

        advantage = returns - tf.stop_gradient(old_action_value)
        self.metrics.add(
            "explained-variance",
            utils.explained_variance(old_action_value, returns), ["mean"], "loss", epoch=True, total=True)

        return advantage

    def action_value_loss(self, action_value, advantage, returns, **kwargs):
        """
        The action_value loss is the MSE of discounted reward and predicted
        :param returns:
        :param predicted:
        :return:
        """
        if self.args["value_loss"] == "huber":

            loss = tf.losses.Huber()(tf.stop_gradient(returns), action_value)

        elif self.args["value_loss"] == "mse":
            loss = tf.losses.mean_squared_error(tf.stop_gradient(returns), action_value)
        else:
            raise NotImplementedError("The loss %s is not implemented for %s." % (self.value_loss, self.name))

        return self.args["value_coef"] * loss
