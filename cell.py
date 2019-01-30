import cell_types


class Cell:
    CELL_DIMENSION = 1000  # X centimeters in width

    def __init__(self, grid, x, y):
        self.grid = grid
        self.type = cell_types.EmptyPoint
        self._occupant = None
        self.x = x
        self.y = y
        self.i = self.x * self.grid.height + self.y

    @property
    def occupant(self):
        return self._occupant

    @occupant.setter
    def occupant(self, x):
        self._occupant = x
        for cb in self.grid.cb_on_cell_change:
            cb(self)

