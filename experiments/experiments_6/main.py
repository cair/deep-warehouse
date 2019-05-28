#!/usr/bin/python

import numpy as np
import gym
import pyximport; pyximport.install(setup_args={
    "include_dirs":np.get_include()},
    reload_support=True)
import MultiClassTsetlinMachine

# Ensembles

ensemble_size = 1000

# Parameters for the Tsetlin Machine
T = 10
s = 3.0
number_of_clauses = 300
states = 100


# Training configuration
epochs = 500

# Loading of training and test data
data = np.loadtxt("BinaryIrisData.txt").astype(dtype=np.int32)

accuracy_training = np.zeros(ensemble_size)
accuracy_test = np.zeros(ensemble_size)

env = gym.make("CartPole-v0")


def preprocess(state):
    i1 = 1 if -2.4 < state[0] < 2.4 else 0
    i2 = 0
    i3 = 1 if -0.41 < state[2] < 0.41 else 0
    i4 = 0
    return [i1, i2, i3, i4]

print(env.reset())

# Parameters of the pattern recognition problem
number_of_features = env.observation_space.shape[0]
number_of_classes = env.action_space.n



for ensemble in range(ensemble_size):
    print("ENSEMBLE", ensemble + 1)

    np.random.shuffle(data)

    X_training = data[:int(data.shape[0]*0.8),0:16] # Input features
    y_training = data[:int(data.shape[0]*0.8),16] # Target value
    print(X_training)
    X_test = data[int(data.shape[0]*0.8):,0:16] # Input features
    y_test = data[int(data.shape[0]*0.8):,16] # Target value

    # This is a multiclass variant of the Tsetlin Machine, capable of distinguishing between multiple classes
    tsetlin_machine = MultiClassTsetlinMachine.MultiClassTsetlinMachine(
        number_of_classes,
        number_of_clauses,
        number_of_features,
        states,
        s,
        T,
        boost_true_positive_feedback=1
    )

    # Training of the Tsetlin Machine in batch mode. The Tsetlin Machine can also be trained online
    tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)

    # Some performacne statistics
    accuracy_test[ensemble] = tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0])
    accuracy_training[ensemble] = tsetlin_machine.evaluate(X_training, y_training, y_training.shape[0])

    print("Average accuracy on test data: %.1f +/- %.1f" % (np.mean(100 * accuracy_test[:ensemble + 1]),
                                                            1.96 * np.std(100 * accuracy_test[:ensemble + 1]) / np.sqrt(
                                                                ensemble + 1)))
    print("Average accuracy on training data: %.1f +/- %.1f" % (np.mean(100 * accuracy_training[:ensemble + 1]),
                                                                1.96 * np.std(
                                                                    100 * accuracy_training[:ensemble + 1]) / np.sqrt(
                                                                    ensemble + 1)))
