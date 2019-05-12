import tensorflow as tf
from tensorflow.python.training.tracking.data_structures import _DictWrapper

from experiments.experiment_5.per_rl import utils

class Policy(tf.keras.models.Model):
    DEFAULTS = dict()
    arguments = utils.arguments

    def __init__(self, agent,
                 dual,
                 update,
                 optimizer=tf.keras.optimizers.Adam(lr=0.001),
                 **kwargs):
        super(Policy, self).__init__()
        self.agent = agent
        self.dual = dual
        self.optimizer = optimizer

        self.trainers = []
        self.update = update
        self.i = 0

        if dual:
            thiscls = self.__class__
            name = self.__class__.__name__
            n_trainers = kwargs["n_trainers"] if "n_trainers" in kwargs else 1
            self.trainers = [("%s_%s" % (n, name), thiscls(agent=agent, dual=not dual, update=update, optimizer=optimizer)) for n in
                             range(n_trainers)]

    def optimize(self, grads):
        self.i += 1
        if self.dual:

            if self.i % self.update["interval"] == 0:
                strategy = self.update["strategy"]

                if strategy == "mean":

                        weights = utils.average_weights([[x * 0.5 for x in trainer.get_weights()] for name, trainer in self.trainers] + [self.get_weights()])
                        self.set_weights(weights)

                elif strategy == "copy":

                    if len(self.trainers) > 1:
                        weights = utils.average_weights([trainer.get_weights() for name, trainer in self.trainers])
                        self.set_weights(weights)
                    else:
                        for name, trainer in self.trainers:
                            self.set_weights(trainer.get_weights())

                else:
                    raise NotImplementedError("The policy update strategy %s is not implemented for the BaseAgent." % strategy)
        else:
            # In the case of multiple optimizers. This also requires the model to define name scopes for variables that should be optimized with that optimizer.
            if isinstance(self.optimizer, _DictWrapper):
                gradvars = {
                    k: [[], []] for k in self.optimizer.keys()
                }

                for grad, var in zip(grads, self.trainable_variables):
                    # Split the variable name
                    varname = var.name.split("/")

                    # The 1th index should be the name scope
                    scope = varname[1]

                    try:
                        gradvars[scope][0].append(grad)
                        gradvars[scope][1].append(var)
                    except KeyError:
                        raise ValueError("Could not find optimizer with matching scope name")

                for scope, grad_vars in gradvars.items():
                    self.optimizer[scope].apply_gradients(zip(*grad_vars))
            else:
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))


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

        self.p_1 = tf.keras.layers.Dense(32, activation="relu", dtype=self.agent.dtype)
        self.p_2 = tf.keras.layers.Dense(32, activation="relu", dtype=self.agent.dtype)
        self.p_3 = tf.keras.layers.Dense(32, activation="relu", dtype=self.agent.dtype)
        self.p_4 = tf.keras.layers.Dense(32, activation="relu", dtype=self.agent.dtype)
        self.logits = tf.keras.layers.Dense(self.agent.action_space, activation="softmax", name='policy_logits')

        self.v_1 = tf.keras.layers.Dense(32, activation="relu", dtype=self.agent.dtype)
        self.v_2 = tf.keras.layers.Dense(32, activation="relu", dtype=self.agent.dtype)
        self.v_3 = tf.keras.layers.Dense(32, activation="relu", dtype=self.agent.dtype)
        self.v_4 = tf.keras.layers.Dense(32, activation="relu", dtype=self.agent.dtype)
        self.action_value = tf.keras.layers.Dense(1,
                                                  activation="linear",
                                                  name="action_value",
                                                  dtype=self.agent.dtype
                                                  )

    def call(self, inputs, **kwargs):

        # Policy Head
        with tf.name_scope("policy"):
            p = self.p_1(inputs)
            p = self.p_2(p)
            p = self.p_3(p)
            p = self.p_4(p)
            policy_logits = self.logits(p)


        # Value Head
        with tf.name_scope("value"):
            v = self.v_1(inputs)
            v = self.v_2(v)
            v = self.v_3(v)
            v = self.v_4(v)
            action_value = self.action_value(v)

        return {
            "logits": policy_logits,
            "action_value": tf.squeeze(action_value)
        }


    """def call(self, inputs, **kwargs):
        # Policy Head
        logits = self.p(inputs)
        action_value = self.v(inputs)

        return {
            "logits": logits,
            "action_value": tf.squeeze(action_value)
        }"""
