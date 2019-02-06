
from deep_logistics import cell_types


class SpawnPoints:

    def __init__(self, environment, seed=None):
        self.environment = environment
        self.data = self.generate

    @property
    def generate(self):
        width = [x for x in range(self.environment.width)]
        height = [0, 1, 2]

        data = []
        for y in height:
            for x in width:
                cell = self.environment.grid.cell(x, y)
                cell.type = cell_types.SpawnPoint
                cell.original_type = cell_types.SpawnPoint
                data.append(cell)

        return data

    def get_available(self):
        return [cell for cell in self.data if not cell.occupant]
