
import time
import random

import numpy as np

import cell_types


class DeliveryPointGenerator:

    def __init__(self, environment, seed=int(time.time()), frequency=0.009):
        self.random = random.Random(seed)
        self.environment = environment
        self.frequency = frequency
        self.data = self.generate

    def roll(self, items, frequency):
        for item in items:
            if self.random.uniform(0, 1) <= frequency:
                yield item

    @property
    def generate(self):
        delivery_points = []
        n_rows = self.environment.h
        n_cols = self.environment.w

        cols = [x for x in range(n_cols)]

        for row in range(1, n_rows):
            selected_cols = list(self.roll(cols, self.frequency))

            for col in selected_cols:
                cell = self.environment.grid[row, col]
                cell.type = cell_types.DeliveryPoint

                delivery_points.append(cell)
        return delivery_points






