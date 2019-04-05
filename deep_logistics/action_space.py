import random


class ActionSpace:
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    NOOP = 4

    DIRECTIONS = {
        LEFT: [-1, 0],
        RIGHT: [+1, 0],
        UP: [0, -1],
        DOWN: [0, 1]
    }

    PRINT = "0:Left, 1:Right, 2:Up, 3:Down, 4:Accelerate, 5:Deaccelerate, 6:Noop"
    N_ACTIONS = 5  # Must be kept up to date with the above.

    @staticmethod
    def sample():
        return random.randint(0, ActionSpace.N_ACTIONS)

    n = 5
