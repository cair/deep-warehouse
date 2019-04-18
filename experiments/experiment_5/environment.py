

class MultiAgentEnvironment:
    pass


class SingleAgentEnvironment:
    pass


class Runner:

    def __init__(self, episodes, callgraph=False):
        self.episodes = episodes
        self.debug_callgraph = ...

    def run(self):
        pass
