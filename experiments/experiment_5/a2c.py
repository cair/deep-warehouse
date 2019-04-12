from experiments.experiment_5.agent import Agent
from experiments.experiment_5.network import PGPolicy
from experiments.experiment_5.pg import REINFORCE
import tensorflow as tf


class A2CPolicy(PGPolicy):

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
            target=lambda agent: A2CPolicy(
                agent=agent,
                inference=True,
                training=True,
                optimizer=tf.keras.optimizers.Adam(lr=0.001)
            )
        )
    )

    def __init__(self,
                 value_coef=0.5,  # For action_value_loss, we multiply by this factor
                 value_loss="huber",
                 entropy_coef=0.01,
                 **kwargs):
        super(A2C, self).__init__(**Agent.arguments())
        self.value_coef = value_coef
        self.value_loss = value_loss
        self.entropy_coef = entropy_coef



        self.add_loss(
            "action_value_loss",
            lambda pred: self.action_value_loss(
                self.G(
                    self.batch.b_rew,
                    self.batch.b_term
                ),
                pred["action_value"]
            )
        )

        if entropy_coef == 0:
            self.add_loss(
                "entropy_loss",
                lambda pred: self.entropy_loss(
                    pred["policy_logits"]
                )
            )

    def G(self, rewards, terminals):
        R = super().G(rewards, terminals)
        V1 = self.predict(self.batch.b_obs1)["action_value"]
        V = self.predict(self.batch.b_obs)["action_value"]

        return R + (V1 * self.gamma - V)

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

        return (entropy_loss * self.entropy_coef)
