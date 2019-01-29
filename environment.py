import asyncio
from multiprocessing import Process, Array
from threading import Thread
import numpy as np
import uvloop
from action_space import ActionSpace
from agent import Agent
from delivery_points import DeliveryPointGenerator
from graphics import Graphics
from grid import Grid
from scheduler import RandomScheduler
from spawn_points import SpawnPoints

class Environment(Process):
    """
    Environment Class, Remember that Numpy Operates with arr[y, x]
    """
    def __init__(self, height, width, depth, agents=1, agent_class=Agent, renderer=None, tile_height=32, tile_width=32,
                 scheduler=RandomScheduler):
        super().__init__()
        self.width = width
        self.height = height
        self.depth = depth
        self.action_space = ActionSpace
        self.renderer = renderer

        """Internal asyncio loop."""
        self.loop = uvloop.new_event_loop()

        """The grid is the global internal state of all cells in the environment."""
        self.grid = Grid(width=width, height=height)

        """Spawn-points is the location where agent can spawn."""
        self.spawn_points = SpawnPoints(self, seed=123)

        """Delivery points is a (TODO) static definition for where agents can deliver scheduled tasks."""
        self.delivery_points = DeliveryPointGenerator(self, seed=123)

        """The scheduler is a engine for scheduling tasks to agents."""
        self.scheduler = scheduler(self)

        """List of all available agents."""
        self.agents = [agent_class(self) for _ in range(agents)]

        """Current selected agent."""
        self.agent = None

        """GUIComponents is a subclass used for rending the internal state of the environment."""
        self.graphics = Graphics(environment=self,
                                 game_width=self.width, game_height=self.height, cell_width=32, cell_height=32)

        """Setup tasks."""
        self.loop.create_task(self.deploy_agents())
        self.loop.create_task(self.scheduler.generator.generate())
        self.loop.create_task(self.task_assignment())
        task_thread = Thread(target=self.loop.run_forever)
        task_thread.start()

    def add_agent(self, agent_cls):
        idx = len(self.agents)
        self.agents.append(agent_cls(self))
        return self.agents[idx]

    def set_agent(self, agent):
        """
        Sets a specific agent to be main environment pov.
        :param agent:
        :return:
        """
        self.agent = agent

    def step(self, action):
        self.agent.do_action(action=action)

    def update(self):
        for agent in self.agents:
            agent.update()

    def render(self):
        for agent in self.agents:
            self.graphics.draw_agent(agent)
            if agent.task:
                self.graphics.draw_pickup_point(agent.task.order_x, agent.task.order_y)
                self.graphics.draw_delivery_point(agent.task.delivery_x, agent.task.delivery_y)

    async def deploy_agents(self):

        """
        Deploy agent if there are any in the queue.
        :return:
        """
        for agent in self.agents:
            if agent.state != Agent.INACTIVE:
                continue

            spawn_points = self.spawn_points.get_available()
            if len(spawn_points) == 0:
                """No available spawn points. """
                continue

            spawn_point = np.random.choice(spawn_points)
            agent.spawn(spawn_point)
            await asyncio.sleep(1)

    async def task_assignment(self):
        """
        Task Assignment is a coroutine which hand_out tasks to free agents. The scheduler can be implemented using various algorithms.
        :return:
        """

        while True:

            for agent in self.agents:
                if agent.task or agent.state == Agent.INACTIVE:
                    """Agent already as a task assigned."""
                    continue

                agent.task = self.scheduler.give_task()
                agent.task.assignee = agent
            await asyncio.sleep(.2)

    def get_shared_state_pointer(self):
        return self.graphics._shared_pointer, self.graphics._shared_pointer_dimensions