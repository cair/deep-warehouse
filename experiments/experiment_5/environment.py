import time

import gym
from gym_deep_logistics import gym_deep_logistics

import ray
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
        return self.state, self.reward, self.terminal, self.info

    def reset(self):
        return self.env.reset()


class Agent:

    def __init__(self, algorithm, algorithm_config, environment, num_agents, num_environments, sample_delay):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.environment = environment
        self.num_agents = num_agents
        self.num_environments = num_environments
        self.sample_delay = sample_delay
        self.task_queue_size = 5

        self.total_steps = 0

        self.remotes = [dict(
            actor=self.createActor(),
            tasks=[],
            accumulated_steps=0
        ) for _ in range(self.num_environments + 1)]
        self.local = self.remotes.pop()

    def createActor(self):
        return EnvActor.remote(self.environment, self.algorithm, self.algorithm_config)

    def train(self):

        for remote in self.remotes:
            actor = remote["actor"]
            tasks = remote["tasks"]

            # Create new tasks if there is few in queue.
            if len(tasks) < self.task_queue_size:
                n_new = self.task_queue_size - len(tasks)
                tasks.extend([actor.train.remote() for _ in range(n_new)])

            completed_ids, _ = ray.wait(tasks, self.task_queue_size, timeout=.01)
            for completed_id in completed_ids:
                episode_steps = ray.get(completed_id)  # the completed task should return number of steps performed
                # during that episode.

                self.total_steps += episode_steps
                remote["accumulated_steps"] += episode_steps

                tasks.remove(completed_id)
            #print("Completed: ", str(len(completed_ids)), "| Incomplete: ", str(len(_)))


        local_actor = self.local["actor"]

        time.sleep(.5)


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

    def get_weights(self):
        pass

    def set_weights(self):
        pass

    def train(self):
        return self.run_episode()

    def run_episode(self):
        steps = 0
        self.env.reset()
        terminal = False
        while not terminal:
            action = self.agent.predict(self.env.state)
            state1, reward, terminal = self.env.step(action)
            self.agent.observe(
                obs1=state1,
                reward=reward,
                terminal=terminal
            )
            steps += 1
        return steps

