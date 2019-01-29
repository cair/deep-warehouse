import random


class ActionSpace:
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    NOOP = 6

    PRINT = "0:Left, 1:Right, 2:Up, 3:Down, 4:Accelerate, 5:Deaccelerate, 6:Noop"
    N_ACTIONS = 7  # Must be kept up to date with the above.

    @staticmethod
    def sample():
        return random.randint(0, ActionSpace.N_ACTIONS)
