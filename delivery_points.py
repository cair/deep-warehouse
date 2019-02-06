
import time
import random

import numpy as np

from deep_logistics import cell_types


class DeliveryPointGenerator:

    def __init__(self, environment, override=None, seed=int(time.time()), frequency=0.04):
        self.random = random.Random(seed)
        self.environment = environment
        self.frequency = frequency

        if override is not None:
            self.data = [self.add_delivery_point(*data) for data in override]
        else:
            self.data = self.generate()

    def roll(self, items, frequency):
        for item in items:
            if self.random.uniform(0, 1) <= frequency:
                yield item

    def add_delivery_point(self, x, y):
        cell = self.environment.grid.cell(x, y)
        cell.type = cell_types.OrderDelivery
        cell.original_type = cell_types.OrderDelivery
        return cell

    def generate(self):
        delivery_points = []

        possible_xs = np.arange(self.environment.width)

        for y in np.arange(self.environment.height):
            if y < 2:
                continue
            selected_xs = list(self.roll(possible_xs, self.frequency))

            for x in selected_xs:
                delivery_points.append(self.add_delivery_point(x, y))

        return delivery_points






