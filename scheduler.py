import abc
import asyncio
import random
import uuid
from collections import namedtuple

import cell_types


class Order:
    Coordinate = namedtuple("Coordinate", ["x", "y", "z"])

    def __init__(self, environment, order_x, order_y, depth, delivery_x, delivery_y):
        self.id = str(uuid.uuid4())
        self.environment = environment

        self.agent = None

        self.x_0 = order_x
        self.y_0 = order_y
        self.z_0 = depth

        self.x_1 = delivery_x
        self.y_1 = delivery_y
        self.z_1 = 0

        self.has_picked_up = False
        self.has_finished = False
        self.has_started = False

        self.c_0 = Order.Coordinate(x=self.x_0, y=self.y_0, z=self.z_0)
        self.c_1 = Order.Coordinate(x=self.x_1, y=self.y_1, z=self.z_1)

    def get_coordinates(self):
        return self.c_1 if self.has_picked_up else self.c_0

    def start(self):
        self.has_started = True

        """Set Target cell to pickup and destination to delivery"""
        cell_0 = self.environment.grid.cell(self.x_0, self.y_0)
        cell_0.order_type = cell_types.OrderPickup
        cell_0.update_type()
        cell_0.trigger_callback()

        cell_1 = self.environment.grid.cell(self.x_1, self.y_1)
        cell_1.order_type = cell_types.OrderDeliveryActive
        cell_1.update_type()
        cell_1.trigger_callback()

    def abort(self):
        self.has_started = False
        self.environment.scheduler.generator.queue.append(self)
        self.agent = None

        """Set Target cell to pickup and destination to delivery"""
        cell_0 = self.environment.grid.cell(self.x_0, self.y_0)
        cell_0.update_type(reset=True)
        cell_0.trigger_callback()

        cell_1 = self.environment.grid.cell(self.x_1, self.y_1)
        cell_1.update_type(reset=True)
        cell_1.trigger_callback()

    def evaluate(self):
        assert self.has_started
        coords = self.get_coordinates()
        at_location = self.agent.cell.x == coords.x and self.agent.cell.y == coords.y

        if not at_location:
            return

        if self.has_picked_up:
            self.has_finished = True
            self.agent.task = None
            self.agent = None

            cell_1 = self.environment.grid.cell(self.x_1, self.y_1)
            cell_1.update_type(reset=True)
            cell_1.trigger_callback()

        else:
            self.has_picked_up = True

            cell_0 = self.environment.grid.cell(self.x_0, self.y_0)
            cell_0.update_type(reset=True)
            cell_0.trigger_callback()

class OrderGenerator:

    def __init__(self, environment, task_frequency=.05, task_init_size=1000):
        self.environment = environment
        self.order_history = []
        self.queue = list()
        self.task_frequency = task_frequency
        self.task_init_size = task_init_size
        self.generate(init=True)

    def generate(self, init=False):
        if init:
            for _ in range(self.task_init_size):
                self.add_task()

        self.add_task()

    def add_task(self):
        x = random.randint(0, self.environment.width - 1)
        y = random.randint(2, self.environment.height - 1)
        depth = random.randint(0, self.environment.depth)

        delivery_point = random.choice(self.environment.delivery_points.data)

        order = Order(self.environment, x, y, depth, delivery_point.x, delivery_point.y)
        self.queue.append(order)


class Scheduler(abc.ABC):

    def __init__(self, environment):
        self.environment = environment
        self.generator = OrderGenerator(environment=environment)

    def give_task(self, agent):
        raise NotImplemented("The give_task function must be implemented in an non abstract version. Example: "
                             "RandomScheduler or DistanceScheduler")


class RandomScheduler(Scheduler):

    def give_task(self, agent):
        if len(self.generator.queue) == 0:
            return None

        pop_at = random.randint(0, len(self.generator.queue) - 1)
        task = self.generator.queue.pop(pop_at)

        if not task:
            """No available task."""
            return
        agent.task = task
        agent.task.agent = agent
        task.start()
