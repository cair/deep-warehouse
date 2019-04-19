import time
import ray
import asyncio
import gym


@ray.remote
class EnvActor:

    def __init__(self, env, agent, agent_config):
        """1. If env is a string, use as gym environment. If its a class, use "as is"."""
        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env = env

        """Agent Class."""
        self._agent_class = agent

        """Agent Configuration"""
        self.agent_config = agent_config

        """Initialize Agent"""
        self.agent = self._agent_class(**self.agent_config)

        self.episodes = 1000

        self.run()

    def run(self):

        for e in range(self.episodes):

            steps = 0
            terminal = False
            obs = self.env.reset()

            while not terminal:
                action = self.agent.get_action(obs[None, :])
                obs, reward, terminal, info = self.env.step(action)
                reward = 0 if terminal else reward
                self.agent.observe(obs[None, :], reward, terminal)
                steps += 1


class Runner:

    def __init__(self):
        self.loop = asyncio.get_event_loop()
        self.actor_ids = []

    def setup(self, env, agent, agent_config, num_actors=1):

        for i in range(num_actors):
            actor_id = EnvActor.remote(env, agent, agent_config)
            self.actor_ids.append(actor_id)

        self.loop.run_forever()

