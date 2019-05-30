#!/usr/bin/python
import struct
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


def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')


def bin_to_float(binary):
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


def preprocess(state):
    bits = []
    for feature in state:
        bits.extend(float_to_bin(feature))

    return np.squeeze(np.asarray([bits], dtype=np.int32))


# Parameters of the pattern recognition problem
number_of_features = len(preprocess(env.reset()))
number_of_classes = env.action_space.n

X_training = data[:int(data.shape[0]*0.8),0:16] # Input features
y_training = data[:int(data.shape[0]*0.8),16] # Target value
print(X_training.shape, y_training.shape)

for ensemble in range(ensemble_size):

    # This is a multiclass variant of the Tsetlin Machine, capable of distinguishing between multiple classes
    tm = MultiClassTsetlinMachine.MultiClassTsetlinMachine(
        number_of_classes,
        number_of_clauses,
        number_of_features,
        states,
        s,
        T,
        boost_true_positive_feedback=1
    )



    for episode in range(1000):
        X_batch = []
        Y_batch = []


        cum_reward = 0
        terminal = False
        obs = env.reset()

        while not terminal:
            obs = preprocess(obs)
            action = tm.predict(obs)

            obs, reward, terminal, _ = env.step(action)
            cum_reward += reward

            X_batch.append(obs)
            Y_batch.append(-1 if terminal else 1)

        X = np.asarray(X_batch, dtype=np.int32)
        Y = np.asarray(Y_batch, dtype=np.int32)
        #print(Y_batch)
        tm.fit(X, Y, Y.shape[0], epochs=epochs)

        print(cum_reward)






    # Training of the Tsetlin Machine in batch mode. The Tsetlin Machine can also be trained online
    #tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)

    # Some performacne statistics
    """accuracy_test[ensemble] = tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0])
    accuracy_training[ensemble] = tsetlin_machine.evaluate(X_training, y_training, y_training.shape[0])

    print("Average accuracy on test data: %.1f +/- %.1f" % (np.mean(100 * accuracy_test[:ensemble + 1]),
                                                            1.96 * np.std(100 * accuracy_test[:ensemble + 1]) / np.sqrt(
                                                                ensemble + 1)))
    print("Average accuracy on training data: %.1f +/- %.1f" % (np.mean(100 * accuracy_training[:ensemble + 1]),
                                                                1.96 * np.std(
                                                                    100 * accuracy_training[:ensemble + 1]) / np.sqrt(
                                                                    ensemble + 1)))"""
