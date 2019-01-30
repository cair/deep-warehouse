import abc
import asyncio
import random
import uuid
from collections import namedtuple


class Order:
    def __init__(self, order_x, order_y, depth, delivery_x, delivery_y):
        self.id = str(uuid.uuid4())
        self.order_x = order_x
        self.order_y = order_y
        self.delivery_x = delivery_x
        self.delivery_y = delivery_y

        self.picked_up = False
        self.done = False

        self.depth = depth
        self.assignee = None
        self.coords = namedtuple("Coordinates", ["x", "y"])
        self.order_coords = self.coords(x=order_x, y=order_y)
        self.delivery_coords = self.coords(x=delivery_x, y=delivery_y)

    def get_coordinates(self):
        if self.picked_up:
            return self.delivery_coords

        return self.order_coords

    def at_location(self):
        coords = self.get_coordinates()
        return self.assignee.cell.x == coords.x and self.assignee.cell.y == coords.y

    def signal(self):
        if self.picked_up:
            self.done = True
            self.assignee.task = None
            self.assignee = None
        else:
            self.picked_up = True


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

        order = Order(x, y, depth, delivery_point.x, delivery_point.y)
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
        agent.task.assignee = agent

