from experiments.experiment_5.network import PGPolicy
from experiments.experiment_5.pg import REINFORCE
import gym
import tensorflow as tf


class A2CPolicy(PGPolicy):

    def __init__(self, action_space, dtype):
        super().__init__(action_space, dtype)

        self.state_value = tf.keras.layers.Dense(1,
                                                 activation="linear",
                                                 name="state_value",
                                                 dtype=dtype
                                                 )

    def call(self, inputs):
        data = super().call(inputs)
        data.update(dict(
            action_value=self.state_value(self.shared(inputs))
        ))
        return data


class A2C(REINFORCE):

    def __init__(self,
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 gamma=0.99,
                 batch_size=1,
                 dtype=tf.float32,
                 tensorboard_enabled=True,
                 tensorboard_path="./tb/",
                 name_prefix=""
                 ):
        super(A2C, self).__init__(
            obs_space=obs_space,
            action_space=action_space,
            gamma=gamma,
            batch_size=batch_size,
            dtype=dtype,
            policy=A2CPolicy(
              action_space=action_space,
              dtype=dtype
            ),
            tensorboard_enabled=tensorboard_enabled,
            tensorboard_path=tensorboard_path,
            name_prefix=name_prefix
        )

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

    def G(self, rewards, terminals):
        R = super().G(rewards, terminals)
        V1 = self.predict(self.batch.b_obs1)["action_value"]
        V = self.predict(self.batch.b_obs)["action_value"]

        return R + (V1*self.gamma - V)

    def action_value_loss(self, returns, predicted):
        """
        The action_value loss is the MSE of discounted reward and predicted
        :param returns:
        :param predicted:
        :return:
        """
        loss = tf.keras.losses.mean_squared_error(returns, predicted)
        loss = tf.keras.backend.mean(loss)
        # TODO loss must be logged.
        return loss
