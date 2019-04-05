import random
from deep_logistics.agent import Agent

class RandomAgent:

    def __init__(self, env):
        self.env = env

    def act(self, states):
        return random.randint(0, self.env.env.action_space.N_ACTIONS - 1)

    def observe(self, reward, terminal):
        pass

class ManhattanAgent:

    def __init__(self, env):
        pass


class AIAgent(Agent):

    def __init__(self, env):
        super().__init__(env)

