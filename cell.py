from deep_logistics import cell_types


class Cell:
    CELL_DIMENSION = 1000  # X centimeters in width

    def __init__(self, grid, x, y):
        self.grid = grid
        self.order_type = None
        self.original_type = cell_types.Empty
        self.type = cell_types.Empty
        self._occupant = None
        self.x = x
        self.y = y
        self.i = self.x * self.grid.height + self.y

    def update_type(self, reset=False):
        if reset:
            self.type = self.original_type
            self.order_type = None
            return

        if self.order_type:
            self.type = self.order_type
            return

        assert False

    @property
    def occupant(self):
        return self._occupant

    @occupant.setter
    def occupant(self, x):
        self._occupant = x
        self.trigger_callback()

    def trigger_callback(self):
        for cb in self.grid.cb_on_cell_change:
            cb(self)
