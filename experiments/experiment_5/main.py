# Multi gpu:
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py

import time

import ray
import gym

from experiments.experiment_5.per_rl.agents.ppo import PPO
from gym_deep_logistics import gym_deep_logistics
from experiments.experiment_5.environment import Agent
from experiments.experiment_5.per_rl.agents.a2c import A2C

if __name__ == "__main__":
    ray.init()

    env_name = "CartPole-v0"
    #env_name = "deep-logistics-normal-v0"
    env = gym.make(env_name)

    agent = Agent(
        algorithm=PPO,
        algorithm_config=dict(
            obs_space=env.observation_space,
            action_space=env.action_space,
            tensorboard_enabled=True,
        ),
        environment=env_name,
        num_agents=1,
        num_environments=30,
        sample_delay=30
    )

    #agent.single_train()

    while True:
        agent.train()
