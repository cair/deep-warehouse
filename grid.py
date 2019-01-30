import numpy as np

from cell import Cell


class Grid:
    MOVE_OK = 1
    MOVE_AGENT_COLLISION = 2
    MOVE_WALL_COLLISION = 3

    def __init__(self, width, height):
        self.grid = np.ndarray(shape=(height, width), dtype=np.object)
        self.width = width
        self.height = height
        for (y, x), index in np.ndenumerate(self.grid):
                self.grid[y, x] = Cell(x=x, y=y)

    def cell(self, x, y):
        return self.grid[y, x]

    def move_relative(self, agent, x, y):
        return self.move(agent, agent.cell.x + x, agent.cell.y + y)

    def move(self, agent, x, y):

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return Grid.MOVE_WALL_COLLISION

        cell = self.grid[y, x]

        if cell.occupant and cell.occupant != agent:
            return Grid.MOVE_AGENT_COLLISION
        else:
            if agent.cell:
                agent.cell.occupant = None
            cell.occupant = agent
            agent.cell = cell
            return Grid.MOVE_OK

    def has_occupant(self,x, y):
        return self.grid[y, x].occupant
