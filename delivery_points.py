
import time
import random

import numpy as np

import cell_types


class DeliveryPointGenerator:

    def __init__(self, environment, seed=int(time.time()), frequency=0.04):
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

        possible_xs = np.arange(self.environment.width)

        for y in np.arange(self.environment.height):
            if y < 2:
                continue
            selected_xs = list(self.roll(possible_xs, self.frequency))

            for x in selected_xs:
                cell = self.environment.grid.cell(x, y)
                cell.type = cell_types.OrderDelivery
                cell.original_type = cell_types.OrderDelivery
                delivery_points.append(cell)

        return delivery_points






