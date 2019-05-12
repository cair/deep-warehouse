import tensorflow as tf
from experiments.experiment_5.per_rl import utils


class Policy:
    DEFAULTS = dict()
    arguments = utils.arguments

    def __init__(self, agent,
                 dual,
                 optimizer=tf.keras.optimizers.Adam(lr=0.001), **kwargs):
        super(Policy, self).__init__()
        self.agent = agent
        self.dual = dual
        self.optimizer = optimizer  # TODO
        self.trainers = []
        self.models = []

        if dual:
            thiscls = self.__class__
            name = self.__class__.__name__
            n_trainers = kwargs["n_trainers"] if "n_trainers" in kwargs else 1
            self.trainers = [("%s_%s" % (n, name), thiscls(agent=agent, dual=not dual, optimizer=optimizer)) for n in
                             range(n_trainers)]

    def add_model(self, name, layers):

        outputs = []
        build_layers = []
        for layer in layers:
            if isinstance(layer, list):
                outputs.append(layer[1])
                build_layers.append(layer[0])
        self.models.append(
            [outputs, tf.keras.models.Sequential(build_layers)]

        )

    def optimize(self, gradients):

        print(dir(self))
        for gradient in gradients:
            self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))
        # policy.optimizer.apply_gradients(zip(grads, policy.trainable_variables))

    def __call__(self, inputs, **kwargs):
        return self.call(inputs, **kwargs)

    def call(self, inputs, **kwargs):
        print(inputs.shape)
        out = {}
        for outputs, model in self.models:
            pred = model(inputs)
            for i, oname in enumerate(outputs):
                out[oname] = pred[i]

        print(out)
        return out

class PGPolicy(Policy):
    DEFAULTS = dict()

    def __init__(self, **kwargs):
        super(PGPolicy, self).__init__(**kwargs)

        self.h_1 = tf.keras.layers.Dense(64, activation='relu', dtype=self.agent.dtype,
                                         kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.h_2 = tf.keras.layers.Dense(64, activation='relu', dtype=self.agent.dtype,
                                         kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.h_3 = tf.keras.layers.Dense(64, activation='relu', dtype=self.agent.dtype,
                                         kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.logits = tf.keras.layers.Dense(self.agent.action_space, activation="linear", name='policy',
                                            dtype=self.agent.dtype)

    def shared(self, x):
        x = self.h_1(x)
        x = self.h_2(x)
        x = self.h_3(x)
        return x

    def call(self, inputs, **kwargs):
        x = self.shared(inputs)
        x = self.logits(x)
        return dict(
            logits=x
        )


class A2CPolicy(PGPolicy):
    """
    Nice resources:
    Blog: http://steven-anker.nl/blog/?p=184
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.h_4 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.h_5 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.h_6 = tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype)
        self.action_value = tf.keras.layers.Dense(1, dtype=self.agent.dtype)

    def call(self, inputs, **kwargs):
        data = super().call(inputs, **kwargs)

        x = self.shared(inputs)
        x = self.h_4(x)
        x = self.h_5(x)
        x = self.h_6(x)
        action_value = self.action_value(x)

        data["action_value"] = tf.squeeze(action_value)
        return data


class PPOPolicy(Policy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_model(
            name="policy",
            layers=[
                tf.keras.layers.Dense(128, input_shape=self.agent.obs_space.shape),
                tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype),
                tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype),
                tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype),
                [tf.keras.layers.Dense(self.agent.action_space, activation="softmax", dtype=self.agent.dtype), "logits"]
            ]
        )

        self.add_model(
            name="value",
            layers=[
                tf.keras.layers.Dense(128, input_shape=self.agent.obs_space.shape),
                tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype),
                tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype),
                tf.keras.layers.Dense(128, activation="relu", dtype=self.agent.dtype),
                [tf.keras.layers.Dense(1, activation="linear", dtype=self.agent.dtype), "action_value"]
        ])

    """def call(self, inputs, **kwargs):
        # Policy Head
        logits = self.p(inputs)
        action_value = self.v(inputs)

        return {
            "logits": logits,
            "action_value": tf.squeeze(action_value)
        }"""
