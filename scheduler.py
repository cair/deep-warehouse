import abc
import asyncio
import random
import uuid
from collections import namedtuple


class Order:
    def __init__(self, order_y, order_x, depth, delivery_y, delivery_x):
        self.id = str(uuid.uuid4())
        self.order_x = order_x
        self.order_y = order_y
        self.delivery_x = delivery_x
        self.delivery_y = delivery_y

        self.picked_up = False
        self.done = False

        self.depth = depth
        self.assignee = None
        self.coords = namedtuple("Coordinates", ["y", "x"])
        self.order_coords = self.coords(x=order_x, y=order_y)
        self.delivery_coords = self.coords(x=delivery_x, y=delivery_y)

    def get_coordinates(self):
        if self.picked_up:
            return self.delivery_coords

        return self.order_coords

    def at_location(self):
        coords = self.get_coordinates()
        return self.assignee.x == coords.x and self.assignee.y == coords.y

    def signal(self):
        if self.picked_up:
            print("Done")
            self.done = True
            self.assignee.task = None
            self.assignee = None
        else:
            print("Pickup")
            self.picked_up = True


class OrderGenerator:

    def __init__(self, environment, task_frequency=.05, task_init_size=10000):
        self.environment = environment
        self.order_history = []
        self.queue = list()
        self.task_frequency = task_frequency
        self.task_init_size = task_init_size

    async def generate(self):

        for _ in range(self.task_init_size):
            await self.add_task()

        while True:
            await self.add_task()
            await asyncio.sleep(self.task_frequency)

    async def add_task(self):
        e_width = self.environment.w
        e_height = self.environment.h
        e_depth = self.environment.d
        x = random.randint(0, e_width)
        y = random.randint(0, e_height)
        depth = random.randint(0, e_depth)
        delivery_point = random.choice(self.environment.delivery_points.data)

        order = Order(y, x, depth, delivery_point.y, delivery_point.x)
        self.queue.append(order)


class Scheduler(abc.ABC):

    def __init__(self, environment):
        self.environment = environment
        self.generator = OrderGenerator(environment=environment)

    def give_task(self):
        raise NotImplemented("The give_task function must be implemented in an non abstract version. Example: "
                             "RandomScheduler or DistanceScheduler")


class RandomScheduler(Scheduler):

    def give_task(self):
        if len(self.generator.queue) == 0:
            return None
        pop_at = random.randint(0, len(self.generator.queue) - 1)
        return self.generator.queue.pop(pop_at)


