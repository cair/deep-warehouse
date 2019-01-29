import cell_types


class Cell:
    CELL_DIMENSION = 1000  # X centimeters in width

    def __init__(self, y, x):
        self.type = cell_types.EmptyPoint
        self.occupant = None
        self.x = x
        self.y = y
