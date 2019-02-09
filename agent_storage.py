from deep_logistics.agent import Agent


class AgentStore:

    def __init__(self, env):
        self.env = env
        self.agents = []

    def __iter__(self):
        return iter(self.agents)

    def __getitem__(self, item):
        return self.agents[item]

    def __len__(self):
        return len(self.agents)

    def add_agent(self, cls=None, n=1):

        if cls is None:
            cls = Agent

        for i in range(n):
            self.agents.append(cls(self.env))

    def is_terminal(self, agent=None):

        if agent:
            pass

        for agent in self.agents:
            if not agent.is_terminal():
                return False
        return True
