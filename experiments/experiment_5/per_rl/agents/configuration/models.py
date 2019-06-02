import tensorflow as tf
from tensorflow.python.training.tracking.data_structures import _DictWrapper
import numpy as np
from experiments.experiment_5.per_rl import utils


class PolicyManager:

    SUPPORTED_STRATEGY = ["copy", "average"]
    SUPPORTED_SYNCHRONIZE_TYPE = ["weights", "gradients"]
    SUPPORTED_DOUBLE_TYPE = [bool, type(None)]

    def __init__(self, agent, policies):
        self.agent = agent
        self._default_optimizer = tf.keras.optimizers.Adam(lr=3e-3)
        self.default = None
        self.policies = {}

        for k, policy_data in policies.items():
            # Validation
            assert "policy" in policy_data, "The attribute 'policy' must be for the the policy " + k
            assert "double" in policy_data, "The attribute 'double' must be set for the policy " + k
            assert "strategy" in policy_data, "The attribute 'strategy' must be set for the policy '%s' supported: %s " % (k, PolicyManager.SUPPORTED_STRATEGY)
            assert "synchronize" in policy_data, "The attribute 'synchronize' must be set for the policy '%s' supported: %s " % (k, PolicyManager.SUPPORTED_SYNCHRONIZE_TYPE)
            assert "interval" in policy_data, "The attribute 'interval' must be for the the policy " + k

            # Expand data attributes
            default = policy_data["default"] if "default" in policy_data else False
            policy = policy_data["policy"]
            optimizer = policy_data["optimizer"] if "optimizer" else self._default_optimizer
            double = policy_data["double"]
            n_trainers = policy_data["n_trainers"] if "n_trainers" else 1
            strategy = policy_data["strategy"]
            synchronize = policy_data["synchronize"]
            interval = policy_data["interval"]

            self.policies[k] = policy(
                agent=agent,
                alias=k,
                optimizer=optimizer,
                double=double,
                n_trainers=n_trainers,
                strategy=strategy,
                synchronize=synchronize,
                interval=interval
            )

            if self.default is None and default:
                self.default = k

            assert self.default is not None and default, "Logic error. The configuration indicates that two policies are set as default. Currently not supported"

    def __call__(self, *args, **kwargs):
        """
        Predict using master # Correct for current use-case. But this features no special case handling
        :param inputs:
        :return:
        """
        return self.policies[self.default](*args)

    def get_default(self):
        return self.policies[self.default]


class Policy:
    DEFAULTS = dict()
    arguments = utils.arguments

    def __init__(self,
                 agent,
                 alias,
                 optimizer,
                 double,
                 n_trainers,
                 strategy,
                 synchronize,
                 interval,
                 **kwargs
                 ):

        self.model: tf.keras.Model = None

        self.agent = agent
        self.alias = alias
        self.optimizer = optimizer
        self.double = double
        self.n_trainers = n_trainers
        self.strategy = strategy
        self.synchronize = synchronize
        self.sync_interval = interval

        self.step = 0
        self._gradients = None
        self.trainers = self._setup_trainers()

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def __iter__(self):
        return self.trainers.__iter__()

    def call(self, inputs):
        raise NotImplementedError("Call must be implemented!")

    def reset(self):
        self._gradients = None

    def set_gradients(self, grads):
        assert self._gradients is None
        self._gradients = grads

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, w):
        self.model.set_weights(w)

    def _optimize_single_optimizer(self):
        # Only a single optimizer for the whole model
        # Apply the gradients directly.
        self.optimizer.apply_gradients(zip(self._gradients, self.model.trainable_variables))

        # Decay learning rate (if set)
        self.optimizer.lr = self.optimizer.lr - (self.optimizer.lr * self.optimizer.decay)

        # Write learning-rate metrics
        self.agent.metrics.add(self.alias + "/" + "optimizer" + "/learning-rate",
                               self.optimizer.lr.numpy(), ["mean"], "hyper-parameter", epoch=True)

    def _optimize_multi_optimizer(self):
        # Multiple optimizers.
        # Here we use with tf.name_scope("xxx"): round the layers (in forward pass) to define the scope
        # for which each optimizers should optimize for.

        # Define dictionary for grad_vars for each optimizer
        grad_vars = {
            k: [[], []] for k in self.optimizer.keys()
        }

        for grad, var in zip(self._gradients, self.model.trainable_variables):

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

    def _synchronize_gradients(self):
        assert False, "Not really implemented!"
        grads = [slave.grads for slave in self.slaves]

        if self.n_trainers > 1:
            self.grads = utils.average_gradients(grads)
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

    def _synchronize_weights(self):

        if self.strategy == "mean":
            # TODO, borked.
            trainer_weights = [trainer.get_weights() for trainer in self.trainers]
            master_weights = [self.get_weights()]

            weights = utils.average_weights(trainer_weights + master_weights)

            self.set_weights(weights)

        elif self.strategy == "copy":

            if self.n_trainers > 1:
                # TODO borked.
                trainer_weights = [trainer.get_weights() for trainer in self.trainers]
                weights = utils.average_weights(trainer_weights)
                self.set_weights(weights)
            else:

                for trainer in self.trainers:
                    self.set_weights(trainer.get_weights())

        else:
            raise NotImplementedError("The policy update strategy %s is not implemented for the BaseAgent." % self.synchronize)

    def optimize(self):
        # TODO also fix value function! Its decreasing and doesnt make sense at all....
        if self.double:

            # Updates should happen in intervals
            if (self.agent.global_step + 1) % self.sync_interval != 0:
                return

            if self.synchronize == "weights":
                self._synchronize_weights()

            elif self.synchronize == "gradients":
                self._synchronize_gradients()
            else:
                raise NotImplementedError("The strategy update type of '%s' is not implemented. Select 'weights' or "
                                          "'gradients'" % self.synchronize)
            # Increment optimization iterator
            self.step += self.sync_interval
        else:
            # Performed by trainers
            self.do_optimize()
            self.step += 1

    def do_optimize(self):
        # Ensure that gradient is calculated
        assert self._gradients is not None, "Gradients is not set. This means that it has not been calculated yet! This code should run after add_gradients is done!"

        # Optimize either with multiple optimizers, or a single optimizer for the whole model.
        if isinstance(self.optimizer, _DictWrapper):
            self._optimize_multi_optimizer()
        else:
            self._optimize_single_optimizer()

        self._gradients = None

    def _setup_trainers(self):
        """
        Sets up update slaves. Slaves are used to train. When its trained for a while, weights are updated back to the master
        :return:
        """
        trainers = []
        master = self
        if self.double:
            trainers = [
                master.__class__(
                    agent=self.agent,
                    alias="%s/%s" % (self.alias, n + 1),
                    optimizer=self.optimizer,
                    double=False,
                    n_trainers=None,
                    strategy=None,
                    synchronize=None,
                    interval=1
                ) for n in range(self.n_trainers)
            ]
        else:
            trainers.append(master)

        return trainers


class PPOPolicy(Policy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        inputs = tf.keras.Input(shape=self.agent.obs_space.shape)
        x = tf.keras.layers.Dense(128, use_bias=False, activation="relu")(inputs)
        x = tf.keras.layers.Dense(128, use_bias=False, activation="relu")(x)

        value = tf.keras.layers.Dense(
            1,
            #kernel_initializer=tf.initializers.orthogonal(),
            kernel_initializer=tf.initializers.VarianceScaling(scale=1.0),
            name="value",
            activation="linear"
        )(x)

        logits = tf.keras.layers.Dense(
            self.agent.action_space,
            kernel_initializer=tf.initializers.VarianceScaling(scale=0.01),
            use_bias=False,
            activation="linear"
        )(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=[logits, value])

    def call(self, inputs, **kwargs):
        logits, value = self.model(np.asarray(inputs))
        return dict(
            logits=logits,
            values=np.squeeze(value),
        )




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


