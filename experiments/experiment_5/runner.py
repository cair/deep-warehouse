import gym

import os
# Disable CUDA
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from experiments.experiment_5.a2c import A2C
from experiments.experiment_5.pg import REINFORCE

import multiprocessing as mp
import tensorflow as tf

if __name__ == "__main__":
    def submit(args):
        env = gym.make('CartPole-v0')
        tf.config.gpu.set_per_process_memory_growth(True)
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


    benchmark = False

    episodes = 1000
    env = gym.make('CartPole-v0')

    if not benchmark:
        submit((A2C, dict(
            obs_space=env.observation_space,
            action_space=env.action_space.n,
            batch_size=32,  # Important
            tensorboard_enabled=True,
            tensorboard_path="./tb/",
            name_prefix="TDOneStep",
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
            )]
        ]

        with mp.Pool(os.cpu_count() - 1) as p:
            p.map(submit, [x + [episodes] for x in agents])