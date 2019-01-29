
import cell_types


class SpawnPoints:

    def __init__(self, loop, environment, seed=None):
        self.loop = loop
        self.environment = environment
        self.data = self.generate

    @property
    def generate(self):
        rows = [0, 1]
        cols = [x for x in range(self.environment.w)]
        data = []
        for row in rows:
            for col in cols:
                cell = self.environment.grid[row, col]
                cell.type = cell_types.SpawnPoint
                data.append(cell)

        return data

    def get_available(self):
        return [cell for cell in self.data if not cell.occupant]
