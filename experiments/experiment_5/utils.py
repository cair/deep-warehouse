import random
import time

import gym
import tensorflow as tf
import inspect

import collections


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def pollute_namespace(o, kwargs):

    for k, v in kwargs.items():
        setattr(o, k, v)


def arguments():
    """Retrieve locals from caller"""
    frame = inspect.currentframe()
    try:
        args = frame.f_back.f_locals["kwargs"]
    except KeyError:
        args = {}

    try:
        cls = frame.f_back.f_locals["__class__"]
        DEFAULTS = cls.DEFAULTS.copy()
    except KeyError:
        DEFAULTS = {}

    del frame

    args = update(DEFAULTS, args)
    return args

def get_defaults(o, additionally: dict):
    """Retrieve defaults for all of the classes in the inheritance hierarchy"""
    blacklist = ["tensorflow", "keras", "tf"]
    hierarchy = inspect.getmro(o.__class__)[:-1]
    hierarchy = [elem for elem in hierarchy if all(c not in elem.__module__ for c in blacklist)]

    DEFAULTS = {}
    for cls in hierarchy:
        print(cls)
        update(DEFAULTS, cls.DEFAULTS)

    if additionally:
        update(DEFAULTS, additionally)

    return DEFAULTS

def is_gpu_faster(model, env_name):
    env = gym.make(env_name)
    obs = env.reset()
    test_times = 1000

    def test(fn):
        start = time.time()
        for i in range(test_times):
            fn()
        end = time.time()
        return end - start

    def train(model, observations):
        start = time.time()
        with tf.GradientTape() as tape:
            predicted_logits = model(observations)

            predicted_logits = predicted_logits["policy_logits"]
            #loss = tf.keras.losses.categorical_crossentropy(predicted_logits, predicted_logits, from_logits=True)
            loss = tf.reduce_mean(predicted_logits * predicted_logits)

        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return time.time() - start

    devices = {"cpu": tf.device('/cpu:0'), "gpu": tf.device('/gpu:0')}

    for device_type, device in devices.items():
        with device:
            print("Inference: %s - %sms" % (device_type, test(lambda: model(obs[None, :]))))

            for n in [1, 16, 32, 64, 128, 256]:
                print("Training %s: %s - %sms" % (n, device_type, test(lambda: train(model, [obs for _ in range(n)]))))



if __name__ == "__main__":
    #print("--------\ntf.float16")
    #is_gpu_faster(PGPolicy(action_space=2,
    #                        dtype=tf.float16,
    #                        optimizer=tf.keras.optimizers.RMSprop(lr=0.001)), "CartPole-v0")
    #print("--------\ntf.float32")
    #is_gpu_faster(PGPolicy(action_space=2,
    #                        dtype=tf.float32,
    #                        optimizer=tf.keras.optimizers.RMSprop(lr=0.001)), "CartPole-v0")
    print("--------\ntf.float64")
    is_gpu_faster(PGPolicy(action_space=2,
                            dtype=tf.float64,
                            optimizer=tf.keras.optimizers.RMSprop(lr=0.001)), "CartPole-v0")

