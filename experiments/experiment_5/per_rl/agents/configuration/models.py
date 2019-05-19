import tensorflow as tf
from tensorflow.python.training.tracking.data_structures import _DictWrapper

from experiments.experiment_5.per_rl import utils


class PolicyManager:

    def __init__(self,
                 policy,
                 n_trainers=0,
                 double=False,
                 interval=1,
                 strategy="copy",
                 type="weights"
                 ):

        self.double = double
        self.interval = interval  # At which rate updates are performed on the master
        self.strategy = strategy  # Which type of update strategy is used for the update type "weights"
        self.type = type  # The update type is through updating weights directly, or by applying gradient averaging
        self.n_trainers = n_trainers  # Number of trainers
        self.i = 0  # Update counter

        self.master = policy
        self.slaves = []

        if self.double:
            policy_class = policy.__class__
            policy_classname = policy.__class__.__name__

            for n in range(self.n_trainers):
                policy_name = "%s/%s" % (n, policy_classname)

                slave = policy_class(
                    alias=policy_name,
                    agent=policy.agent,
                    optimizer=policy.optimizer
                )

                self.slaves.append(slave)
        else:
            self.slaves.append(self.master)

    def __call__(self, *args, **kwargs):
        """
        Predict using master # Correct for current use-case. But this features no special case handling # TODO?
        :param inputs:
        :return:
        """
        return self.master(*args)

    def _master_optimize_grads(self):

        grads = [slave.grads for slave in self.slaves]

        if self.n_trainers > 1:
            self.master.grads = utils.average_gradients(grads)
        else:
            self.master.grads = grads[0]

        """if self.strategy == "mean":
            # Mean of all slave grads + master then optimize with the mean of grads
            self.master.grads = utils.average_gradients(grads)

        elif self.strategy == "copy":
            # Get all grads for slaves, find mean, then copy over to master

            if self.n_trainers > 1:
                self.master.grads = utils.average_gradients(grads)
            else:
                self.master.grads = grads[0]"""

        self.master.optimize()

    def _master_optimize_weights(self):

        if self.strategy == "mean":

            slave_weights = [slave.get_weights() for slave in self.slaves]
            master_weights = [self.master.get_weights()]

            weights = utils.average_weights(slave_weights + master_weights)

            self.master.set_weights(weights)

        elif self.strategy == "copy":

            if self.n_trainers > 1:

                slave_weights = [slave.get_weights() for slave in self.slaves]
                weights = utils.average_weights(slave_weights)
                self.master.set_weights(weights)
            else:
                for trainer in self.slaves:
                    self.master.set_weights(trainer.get_weights())

        else:
            raise NotImplementedError("The policy update strategy %s is not implemented for the BaseAgent." % self.strategy)

    def _master_optimize(self):

        # Updates should happen in intervals
        if self.i % self.interval != 0:
            return

        if self.type == "weights":
            self._master_optimize_weights()
        elif self.type == "gradients":
            self._master_optimize_grads()
        else:
            raise NotImplementedError("The strategy update type of '%s' is not implemented. Select 'weights' or "
                                      "'gradients'" % self.type)

    def optimize(self):

        # Optimize all slaves
        for slave in self.slaves:
            slave.optimize()

        # In double mode, the master is not optimize, Here we want to copy over at some interval
        if self.double:
            self._master_optimize()


class Policy(tf.keras.models.Model):
    DEFAULTS = dict()
    arguments = utils.arguments

    def __init__(self,
                 agent,
                 alias="root",
                 optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                 ):
        super(Policy, self).__init__()
        self.alias = alias
        self.agent = agent
        self.optimizer = optimizer
        self.i = 0

        self.grads = None

    def reset(self):
        self.grads = None

    def set_grads(self, grads):
        self.grads = grads

    def optimize(self):
        # Increment optimization iterator
        self.i += 1

        # Ensure that gradient is calculated
        if self.grads is None:
            raise RuntimeError("Gradients is not set. This means that it has not been calculated yet! "
                               "This code should run after set_grads is done!")

        if isinstance(self.optimizer, _DictWrapper):
            # Multiple optimizers.
            # Here we use with tf.name_scope("xxx"): round the layers (in forward pass) to define the scope
            # for which each optimizers should optimize for.

            # Define dictionary for grad_vars for each optimizer
            grad_vars = {
                k: [[], []] for k in self.optimizer.keys()
            }

            for grad, var in zip(self.grads, self.trainable_variables):

                # Split the tf variable name
                var_name = var.name.split("/")

                # The 1th index should be the name scope
                scope = var_name[1]

                # Attempt to add grad_var into the scope, if this fails
                # it means that the scope is not covered by the optimizers.
                try:
                    grad_vars[scope][0].append(grad)
                    grad_vars[scope][1].append(var)
                except KeyError:
                    raise ValueError("Could not find optimizer with matching scope name")

            # Iterate over the collected/categorized grad_vars
            for scope, grad_vars in grad_vars.items():

                # Retrieve the optimizer for the selected scope
                optimizer = self.optimizer[scope]

                # Apply gradients
                optimizer.apply_gradients(zip(*grad_vars))

                # Decay learning rate (if set)
                optimizer.lr = optimizer.lr - (optimizer.lr * optimizer.decay)

                # Write learning-rate metrics
                self.agent.metrics.add(self.alias + "/" + scope + "/learning-rate",
                                       optimizer.lr.numpy(), ["mean"], "hyper-parameter", epoch=True)

        else:
            # Only a single optimizer for the whole model
            # Apply the gradients directly.
            self.optimizer.apply_gradients(zip(self.grads, self.trainable_variables))

            # Decay learning rate (if set)
            self.optimizer.lr = self.optimizer.lr - (self.optimizer.lr * self.optimizer.decay)

            # Write learning-rate metrics
            self.agent.metrics.add(self.alias + "/" + "optimizer" + "/learning-rate",
                                   self.optimizer.lr.numpy(), ["mean"], "hyper-parameter", epoch=True)



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

        self.p_1 = tf.keras.layers.Dense(64, activation="tanh", dtype=self.agent.dtype)
        self.p_2 = tf.keras.layers.Dense(64, activation="tanh", dtype=self.agent.dtype)
        self.p_3 = tf.keras.layers.Dense(64, activation="tanh", dtype=self.agent.dtype)
        self.p_4 = tf.keras.layers.Dense(64, activation="tanh", dtype=self.agent.dtype)
        self.logits = tf.keras.layers.Dense(self.agent.action_space,
                                            activation="linear",
                                            name='policy_logits'
                                            )

        self.v_1 = tf.keras.layers.Dense(64, activation="tanh", dtype=self.agent.dtype)
        self.v_2 = tf.keras.layers.Dense(64, activation="tanh", dtype=self.agent.dtype)
        self.v_3 = tf.keras.layers.Dense(64, activation="tanh", dtype=self.agent.dtype)
        self.v_4 = tf.keras.layers.Dense(64, activation="tanh", dtype=self.agent.dtype)
        self.action_value = tf.keras.layers.Dense(1,
                                                  activation="linear",
                                                  name="values",
                                                  dtype=self.agent.dtype)

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
            "values": tf.squeeze(action_value)
        }


    """def call(self, inputs, **kwargs):
        # Policy Head
        logits = self.p(inputs)
        action_value = self.v(inputs)

        return {
            "logits": logits,
            "action_value": tf.squeeze(action_value)
        }"""
