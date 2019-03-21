import abc
from random import Random

from deep_logistics import cell_types


class SpawnStrategy(abc.ABC):

    def __init__(self, environment, seed=None):
        self.env = environment
        self.rnd = Random() if seed is None else Random(x=seed)
        self.data = self.generate()

    def generate(self):
        raise NotImplementedError("generate must be implemented!")

    def get_available(self):
        raise NotImplementedError("get_available must be implemented!")


class RandomSpawnStrategy(SpawnStrategy):
    def generate(self):
        width = [x for x in range(self.env.width)]
        height = [y for y in range(self.env.height)]

        data = []
        for y in height:
            for x in width:
                cell = self.env.grid.cell(x, y)
                #cell.type = cell_types.SpawnPoint
                #cell.original_type = cell_types.SpawnPoint
                data.append(cell)

        return data

    def get_available(self):
        return [cell for cell in self.data if not cell.occupant]


class LocationSpawnStrategy(SpawnStrategy):

    def generate(self):
        width = [x for x in range(self.env.width)]
        height = [0, self.env.height - 1]

        data = []
        for y in height:
            for x in width:
                cell = self.env.grid.cell(x, y)
                cell.type = cell_types.SpawnPoint
                cell.original_type = cell_types.SpawnPoint
                data.append(cell)

        return data

    def get_available(self):
        return [cell for cell in self.data if not cell.occupant]
