import time
import ray
import asyncio
import gym
import numpy as np
import sys
class Environment:

    def __init__(self, env, episodes=sys.maxsize):

        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env = env
            # TODO - Make has attribute checks for step, reset.. etc
        self.episode = 0
        self.max_episode = episodes
        self.steps = 0

        self.state = self.env.reset()
        self.next_state = None
        self.reward = None
        self.terminal = None
        self.info = None

    def step(self, action):
        self.state, self.reward, self.terminal, self.info = self.env.step(action)
        #self.reward = 0 if self.terminal else self.reward # Discounting should handle this anyway.

        self.steps += 1
        if self.terminal:
            self.steps = 0
            self.state = self.env.reset()
            self.episode += 1

        # S, A, R, T
        #state = self.state
        #self.state = self.next_state
        return self.state, self.reward, self.terminal


class Agent:

    def __init__(self, algorithm, algorithm_config, environment, num_agents, num_environments):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.environment = environment
        self.num_agents = num_agents
        self.num_environments = num_environments

        self.remotes = [self.createActor() for _ in range(self.num_environments + 1)]
        self.local = self.remotes.pop()

    def createActor(self):
        return EnvActor.remote(self.environment, self.algorithm, self.algorithm_config)

    def single_train(self):
        for actor in self.remotes:
            batch = actor.single_train.remote()
            print(ray.get(batch))
        self.local.single_train.remote()

@ray.remote
class EnvActor:

    def __init__(self, env, agent, agent_config):
        """1. If env is a string, use as gym environment. If its a class, use "as is"."""
        self.env = Environment(env)

        """Agent Class."""
        self._agent_class = agent

        """Agent Configuration"""
        self.agent_config = agent_config

        """Initialize Agent"""
        self.agent = self._agent_class(**self.agent_config)

        self.episodes = 1000

    def single_train(self):
        self.run_episode()
        return self.agent.batch.data

    def run_episode(self):
        terminal = False
        while not terminal:
            action = self.agent.predict(self.env.state)
            state1, reward, terminal = self.env.step(action)
            self.agent.observe(
                obs1=state1,
                reward=reward,
                terminal=terminal
            )

