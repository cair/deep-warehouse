import time

import gym
from gym_deep_logistics import gym_deep_logistics
import tensorflow as tf
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

        self.local = dict(
            actor=self.createActor(),
            tasks=[],
            accumulated_steps=0
        )
        self.local_task = None

        self.remotes = [dict(
            actor=self.createActor(local=self.local["actor"]),
            tasks=[],
            accumulated_steps=0
        ) for _ in range(self.num_environments)]

    def createActor(self, local=None):
        return EnvActor.remote(self.environment, self.algorithm, self.algorithm_config, local)

    def train(self):

        for remote in self.remotes:
            # Inference on remotes

            actor = remote["actor"]
            tasks = remote["tasks"]

            # Create new tasks if there is few in queue.
            if len(tasks) < self.task_queue_size:
                n_new = self.task_queue_size - len(tasks)
                tasks.extend([actor.train.remote() for _ in range(n_new)])

            completed_ids, _ = ray.wait(tasks, self.task_queue_size, timeout=.01)

            for completed_id in completed_ids:
                episode_steps = ray.get(completed_id)  # the completed task should return number of steps performed
                self.total_steps += episode_steps
                remote["accumulated_steps"] += episode_steps
                tasks.remove(completed_id)

        # Process global worker
        if self.local_task is None:
            self.local_task = self.local["actor"].train.remote()

        local_complete, _ = ray.wait([self.local_task], 1, timeout=.01)
        if len(local_complete) > 0:
            # Training task is done. Update weights on remotes
            self.local_task = None

            if ray.get(local_complete[0]):

                for remote in self.remotes:
                    remote["actor"].set_weights.remote(
                        self.local["actor"].get_weights.remote()
                    )

        time.sleep(.01)



@ray.remote
class EnvActor:

    def __init__(self, env, agent, agent_config, remote=None):
        """1. If env is a string, use as gym environment. If its a class, use "as is"."""
        self.env = Environment(env)
        self.episodes = 1000
        self.remote = remote

        if remote:
            self.has_remote_agent = True
        else:
            self.has_remote_agent = False

        """Agent Class."""
        self._agent_class = agent

        """Initialize Agent"""
        self.agent = self._agent_class(**agent_config)

    def set_weights(self, weights):
        print("Setting weights.")
        self.agent.policy.master.set_weights(weights)
        self.agent.batch.done()
        return True

    def get_weights(self):
        return self.agent.policy.master.get_weights()

    def predict(self, state):
        return self.agent.predict(state)

    def push_batch(self, batch):

        self.agent.batch.extend(batch)

    def train(self):
        return self.run_episode()

    def run_episode(self):
        if self.has_remote_agent:
            return self.explorer()
        else:
            return self.trainer()

    def trainer(self):
        if not self.agent.batch.ready():
            return False
        print("")
        self.agent.train()
        return True

    def explorer(self):
        steps = 0
        self.env.reset()
        terminal = False
        while not terminal:

            action = self.agent.predict(self.env.state)
            state1, reward, terminal, _ = self.env.step(action)
            self.agent.observe(
                actions=tf.one_hot(action, self.agent.action_space),
                rewards=reward,
                terminals=terminal
            )
            steps += 1

            self.handle_batch()

        return steps

    def handle_batch(self):
        if self.has_remote_agent and self.agent.batch.ready():
            self.remote.push_batch.remote(self.agent.batch.get())
            self.agent.batch.done()
