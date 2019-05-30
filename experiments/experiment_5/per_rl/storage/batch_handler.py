import numpy as np


class DynamicBatch:

    def __init__(self, agent, **kwargs):
        self.agent = agent
        self.episodic = self.agent.buffer_mode == "episodic"
        self.buffer_size = 1000000 if self.episodic else self.agent.buffer_size
        self.batch_size = self.agent.batch_size
        self.batch_count = 1 if self.episodic else int(self.buffer_size / self.batch_size)
        self.dtype = agent.dtype

        self.counter = 0
        self.data = {}  # Array of dicts.

    def add(self, **kwargs):

        for k, v in kwargs.items():
            try:
                self.data[k]
            except KeyError:

                self.data[k] = []

            self.data[k].append(np.squeeze(v))

        self.counter += 1
        if self.episodic:
            try:
                return bool(kwargs["terminal"])
            except KeyError:
                raise KeyError("In order to use episodic mode, 'terminal' key must be present in the dataset!")

        return self.counter >= self.buffer_size

    def ready(self):
        return self.counter >= self.buffer_size

    def get(self):
        return {k: np.asarray(v) for k, v in self.data.items()}

    def extend(self, data):
        for k, v in data.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].extend(v)

        self.counter += len(data[list(data.keys())[0]])

    def done(self):
        self.data = {k: [] for k in self.data.keys()}
        self.counter = 0
