import tensorflow as tf

from experiments.experiment_5.per_rl.agents.configuration.models import Policy


class PPOPolicy(Policy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.p_1 = tf.keras.layers.Dense(
            512,
            activation="relu",
            dtype=self.agent.dtype,
            use_bias=False,
        )
        self.p_2 = tf.keras.layers.Dense(
            512,
            activation="relu",
            dtype=self.agent.dtype,
            use_bias=False,
        )
        self.logits = tf.keras.layers.Dense(self.agent.action_space,
                                            dtype=self.agent.dtype,
                                            activation="linear",  # or tanh
                                            name='policy_logits',
                                            use_bias=False,
                                            kernel_initializer=tf.initializers.VarianceScaling(scale=0.01)
                                            )

        #self.v_1 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        #self.v_2 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.action_value = tf.keras.layers.Dense(1,
                                                  activation="linear",
                                                  name="values",
                                                  use_bias=False,
                                                  kernel_initializer=tf.initializers.VarianceScaling(scale=1.0),
                                                  dtype=self.agent.dtype)

    def call(self, inputs, **kwargs):

        # Policy Head
        with tf.name_scope("policy"):
            p = self.p_1(inputs)
            p = self.p_2(p)
            policy_logits = self.logits(p)

        # Value Head
        with tf.name_scope("value"):
            #v = self.v_1(inputs)
            #v = self.v_2(v)
            action_value = self.action_value(p)



        return {
            "logits": policy_logits,
            "values": tf.squeeze(action_value),

        }


    """def call(self, inputs, **kwargs):
        # Policy Head
        logits = self.p(inputs)
        action_value = self.v(inputs)

        return {
            "logits": logits,
            "action_value": tf.squeeze(action_value)
        }"""
