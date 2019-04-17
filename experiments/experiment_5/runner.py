import gym

import os
# Disable CUDA
#
from experiments.experiment_5.a2c import A2C, A2CPolicy
from experiments.experiment_5.pg import REINFORCE

import multiprocessing as mp
import tensorflow as tf

from experiments.experiment_5.ppo import PPO

tf.config.gpu.set_per_process_memory_growth(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":
    benchmark = True
    episodes = 10000
    env_name = "CartPole-v0"

    def submit(args):
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            env = gym.make(env_name)

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
        except Exception as e:
            print(e)
            raise e

    env = gym.make(env_name)

    if not benchmark:
        """submit((REINFORCE, dict(
            obs_space=env.observation_space,
            action_space=env.action_space.n,
            batch_mode="episodic",
            batch_size=64,
            tensorboard_enabled=True,
            tensorboard_path="./tb/"
        ), episodes))"""
        submit((A2C, dict(
            obs_space=env.observation_space,
            action_space=env.action_space.n,
            batch_size=64,  # Important
            batch_mode="steps",
            policies=dict(
                target=lambda agent: A2CPolicy(
                    agent=agent,
                    inference=True,
                    training=True,
                    optimizer=tf.keras.optimizers.RMSprop(lr=0.001)
                )
            ),
            tensorboard_enabled=True,
            tensorboard_path="./tb/",
            name_prefix="SinglePolicy",
        ), episodes))
    else:
        agents = [
            [REINFORCE, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_size=64,
                batch_mode="episodic",
                tensorboard_enabled=True,
                tensorboard_path="./tb/"
            )],
            [A2C, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_mode="episodic",
                batch_size=64,
                tensorboard_enabled=True,
                tensorboard_path="./tb/",
                name_prefix="Episodic"
            )],
            [A2C, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_size=64,  # Important
                batch_mode="steps",
                tensorboard_enabled=True,
                tensorboard_path="./tb/",
                name_prefix="Steps64",
            )],
            [A2C, dict(
                obs_space=env.observation_space,
                action_space=env.action_space.n,
                batch_size=64,  # Important
                batch_mode="steps",
                policies=dict(
                    target=lambda agent: A2CPolicy(
                        agent=agent,
                        inference=True,
                        training=True,
                        optimizer=tf.keras.optimizers.RMSprop(lr=0.001)
                    )
                ),
                tensorboard_enabled=True,
                tensorboard_path="./tb/",
                name_prefix="SinglePolicy",
            )]
        ]

        with mp.Pool(os.cpu_count() ) as p:
            p.map(submit, [x + [episodes] for x in agents])
