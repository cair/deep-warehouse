from experiments.experiment_5 import utils
from experiments.experiment_5.agent import Agent
from experiments.experiment_5.network import PGPolicy
from experiments.experiment_5.pg import REINFORCE
import tensorflow as tf



class A2CPolicy(PGPolicy):
    """
    Nice resources:
    Blog: http://steven-anker.nl/blog/?p=184
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.h_4 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.state_value = tf.keras.layers.Dense(1,
                                                 activation="linear",
                                                 name="state_value",
                                                 dtype=self.agent.dtype
                                                 )

    def call(self, inputs):
        data = super().call(inputs)
        data.update(dict(
            action_value=self.state_value(self.h_4(self.shared(inputs)))
        ))
        return data


class A2C(REINFORCE):

    DEFAULTS = dict(
        batch_mode="steps",
        batch_size=64,
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
                optimizer=tf.keras.optimizers.RMSprop(lr=0.001)
            )
        ),
        policy_update=dict(
            interval=5,  # Update every 5 training epochs,
            strategy="copy",  # "copy, mean"
        )
    )

    def __init__(self,
                 value_coef=0.5,  # For action_value_loss, we multiply by this factor
                 value_loss="huber",
                 entropy_coef=0.0001,
                 **kwargs):
        super(A2C, self).__init__(**Agent.arguments())

        self.value_coef = value_coef
        self.value_loss = value_loss
        self.entropy_coef = entropy_coef

        self.metrics.text("explained_variance", "Explained Variance is an attempt to measure the quality of the state "
                                                "value.  \n"
                                                "**ev=0**: Might as well have predicted zero  \n"
                                                "**ev=1**: Perfect prediction  \n  "
                                                "**ev<0**: Worse than just predicting zero")

        self.add_loss(
            "action_value_loss",
            lambda prediction, data: self.action_value_loss(
                self.discounted_returns(
                    data["rewards"],
                    data["terminals"]
                ),
                prediction["action_value"]
            )
        )

        """This overrides PG loss"""
        self.add_loss("policy_loss",
                      lambda prediction, data: self.policy_loss(
                          data["actions"],
                          self.discounted_returns(
                              data["rewards"],
                              data["terminals"]
                          ),
                          prediction["policy_logits"]
                      ))

        if entropy_coef != 0:
            self.add_loss(
                "entropy_loss",
                lambda prediction, data: self.entropy_loss(
                    prediction["policy_logits"]
                )
            )

    def ADV(self, policy, obs, obs1, rewards, terminals):
        R = super().discounted_returns(policy, obs, obs1, rewards, terminals)

        #V = tf.squeeze(policy(obs)["action_value"])
        #V1 = tf.squeeze(policy(obs1)["action_value"])
        #returns = self.GAE(V, V1, rewards, terminals)

        #A = returns - V

        return 0 # R + ((V1*self.gamma) - V)


    def action_value_loss(self, returns, predicted):
        """
        The action_value loss is the MSE of discounted reward and predicted
        :param returns:
        :param predicted:
        :return:
        """
        if self.value_loss == "huber":
            loss = tf.losses.Huber()
            loss = loss(returns, predicted)
        elif self.value_loss == "mse":
            loss = tf.keras.losses.mean_squared_error(returns, predicted)
        else:
            raise NotImplementedError("The loss %s is not implemented for %s." % (self.value_loss, self.name))
        loss *= self.value_coef

        return loss

    def entropy_loss(self, predicted):
        #a = tf.keras.losses.categorical_crossentropy(predicted, predicted, from_logits=True)
        #a = tf.reduce_sum(a)

        """Entropy loss, according to:
        H(x) = -\sum_{i=1}^n {\mathrm{P}(x_i) \log_e \mathrm{P}(x_i)}
        """
        entropy_loss = -tf.reduce_sum(predicted * tf.math.log(predicted))

        return entropy_loss * self.entropy_coef
