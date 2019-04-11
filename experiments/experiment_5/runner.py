import gym

import os
# Disable CUDA
#
from experiments.experiment_5.a2c import A2C, A2CPolicy
from experiments.experiment_5.pg import REINFORCE

import multiprocessing as mp
import tensorflow as tf
tf.config.gpu.set_per_process_memory_growth(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":
    def submit(args):

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        env = gym.make('CartPole-v0')

        AGENT, spec, episodes = args
        agent = AGENT(**spec)

        for e in range(episodes):

            steps = 0
            terminal = False
            obs = env.reset()
            cum_loss = 0
            loss_n = 0

            while not terminal:
                action = agent.get_action(obs[None, :])
                obs, reward, terminal, info = env.step(action)
                reward = 0 if terminal else reward
                agent.observe(obs[None, :], reward, terminal)
                steps += 1



    benchmark = True

    episodes = 100000
    env = gym.make('CartPole-v0')

    if not benchmark:
        submit((A2C, dict(
            obs_space=env.observation_space,
            action_space=env.action_space.n,
            batch_size=64,  # Important
            tensorboard_enabled=True,
            tensorboard_path="./tb/",
            name_prefix="Huber_64_RMSprop_big",
            policies=dict(
                target=dict(
                    model=A2CPolicy,
                    args=dict(
                        action_space=env.action_space.n,
                        dtype=tf.float32,
                        optimizer=tf.keras.optimizers.RMSprop(lr=0.001)
                    ),
                    training=True,
                    inference=True
                )
            )
        ), episodes))
    else:
        agents = [
            [REINFORCE, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_size=64,
                tensorboard_enabled=True,
                tensorboard_path="./tb/"
            )],
            [REINFORCE, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                baseline="reward_mean",
                batch_size=64,
                tensorboard_enabled=True,
                tensorboard_path="./tb/",
                name_prefix="MeanBaseline"
            )],
            [A2C, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_size=64,  # Important
                tensorboard_enabled=True,
                tensorboard_path="./tb/",
                name_prefix="Huber_64",
            )],
            [A2C, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_size=64,  # Important
                tensorboard_enabled=True,
                tensorboard_path="./tb/",
                name_prefix="Huber_64_RMSprop_small",
                policies=dict(
                    target=dict(
                        model=A2CPolicy,
                        args=dict(
                            action_space=env.action_space.n,
                            dtype=tf.float32,
                            optimizer=tf.keras.optimizers.RMSprop(lr=0.0001)
                        ),
                        training=True,
                        inference=True
                    )
                )
            )],
            [A2C, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_size=64,  # Important
                tensorboard_enabled=True,
                tensorboard_path="./tb/",
                name_prefix="Huber_64_Adam_small",
                policies=dict(
                    target=dict(
                        model=A2CPolicy,
                        args=dict(
                            action_space=env.action_space.n,
                            dtype=tf.float32,
                            optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                        ),
                        training=True,
                        inference=True
                    )
                )
            )],
            [A2C, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_size=64,  # Important
                tensorboard_enabled=True,
                tensorboard_path="./tb/",
                name_prefix="Huber_64_RMSprop_big",
                policies=dict(
                    target=dict(
                        model=A2CPolicy,
                        args=dict(
                            action_space=env.action_space.n,
                            dtype=tf.float32,
                            optimizer=tf.keras.optimizers.RMSprop(lr=0.001)
                        ),
                        training=True,
                        inference=True
                    )
                )
            )],
            [A2C, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_size=64,  # Important
                tensorboard_enabled=True,
                tensorboard_path="./tb/",
                name_prefix="Huber_64_Adam_big",
                policies=dict(
                    target=dict(
                        model=A2CPolicy,
                        args=dict(
                            action_space=env.action_space.n,
                            dtype=tf.float32,
                            optimizer=tf.keras.optimizers.Adam(lr=0.001)
                        ),
                        training=True,
                        inference=True
                    )
                )
            )],
            [A2C, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_size=64,  # Important
                tensorboard_enabled=True,
                tensorboard_path="./tb/",
                name_prefix="Huber_64_RMSprop_small_no_entropy",
                entropy_coef=0,
                policies=dict(
                    target=dict(
                        model=A2CPolicy,
                        args=dict(
                            action_space=env.action_space.n,
                            dtype=tf.float32,
                            optimizer=tf.keras.optimizers.RMSprop(lr=0.0001)
                        ),
                        training=True,
                        inference=True
                    )
                )
            )],
            [A2C, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_size=64,  # Important
                tensorboard_enabled=True,
                tensorboard_path="./tb/",
                name_prefix="Huber_64_Adam_small_no_entropy",
                entropy_coef=0,
                policies=dict(
                    target=dict(
                        model=A2CPolicy,
                        args=dict(
                            action_space=env.action_space.n,
                            dtype=tf.float32,
                            optimizer=tf.keras.optimizers.Adam(lr=0.0001)
                        ),
                        training=True,
                        inference=True
                    )
                )
            )],
        ]

        with mp.Pool(os.cpu_count() - 1) as p:
            p.map(submit, [x + [episodes] for x in agents])