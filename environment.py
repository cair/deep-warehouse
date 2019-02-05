from multiprocessing import Process, Array

import time
import numpy as np

from action_space import ActionSpace
from agent import Agent
from delivery_points import DeliveryPointGenerator
from graphics import Graphics, PygameGraphics
from grid import Grid
from scheduler import RandomScheduler
from spawn_points import SpawnPoints


class Environment(Process):
    """
    Environment Class, Remember that Numpy Operates with arr[y, x]
    """
    def __init__(self,
                 height,
                 width,
                 depth,
                 agents=1,
                 agent_class=Agent,
                 renderer=None,
                 tile_height=32,
                 tile_width=32,
                 ups=None,
                 scheduler=RandomScheduler,
                 ticks_per_second=10,
                 spawn_interval=1,
                 task_generate_interval=5,
                 task_assign_interval=1):
        super().__init__()
        self.width = width
        self.height = height
        self.depth = depth
        self.action_space = ActionSpace
        self.renderer = renderer

        """Updates per second."""
        self.ups = ups
        self.ups_interval = 0 if self.ups is None else 1.0 / self.ups

        """Ticks per second."""
        self.tick_ps = ticks_per_second
        self.tick_ps_ratio = 1 / self.tick_ps
        self.tick_ps_counter = 0

        """Spawn interval in game-seconds."""
        self.spawn_interval = spawn_interval

        """Task assignment interval in game-seconds."""
        self.task_assignment_interval = task_assign_interval

        """Task creation interval in game-seconds"""
        self.task_generate_interval = task_generate_interval  # TODO use this?

        """The grid is the global internal state of all cells in the environment."""
        self.grid = Grid(width=width, height=height)

        """Spawn-points is the location where agent can spawn."""
        self.spawn_points = SpawnPoints(self, seed=12)

        """Delivery points is a (TODO) static definition for where agents can deliver scheduled tasks."""
        self.delivery_points = DeliveryPointGenerator(self, seed=555)

        """The scheduler is a engine for scheduling tasks to agents."""
        self.scheduler = scheduler(self)

        """Current selected agent."""
        self.agent = None

        """List of all available agents."""
        self.agents = []
        for _ in range(agents):
            self.add_agent(agent_cls=agent_class)

        """GUIComponents is a subclass used for rending the internal state of the environment."""
        self.graphics = PygameGraphics(environment=self,
                                 game_width=self.width,
                                 game_height=self.height,
                                 cell_width=tile_width,
                                 cell_height=tile_height
                                 )

    def add_agent(self, agent_cls):
        idx = len(self.agents)
        self.agents.append(agent_cls(self))
        if self.agent is None:
            self.agent = self.agents[idx]

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

    def get_seconds(self):
        return self.tick_ps_counter * self.tick_ps_ratio

    def update(self):
        self.tick_ps_counter += 1
        seconds = self.get_seconds()
        if seconds % self.spawn_interval == 0:
            self.deploy_agents()

        if seconds % self.task_assignment_interval == 0:
            self.task_assignment()

        if seconds % self.task_assignment_interval == 0:
            self.scheduler.generator.generate(init=False)

        for agent in self.agents:
            agent.update()
            """Evaluate task objective."""
            if agent.task:
                agent.task.evaluate()

        if self.ups:
            time.sleep(self.ups_interval)

    def render(self):
        self.graphics.reset()
        for agent in self.agents:
            self.graphics.draw_agent(agent)
            if agent.task:
                self.graphics.draw_pickup_point(agent.task.x_0, agent.task.y_0)
                self.graphics.draw_delivery_point(agent.task.x_1, agent.task.y_1)

        self.graphics.blit()

    def deploy_agents(self):

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

    def task_assignment(self):
        """
        Task Assignment is a coroutine which hand_out tasks to free agents. The scheduler can be implemented using various algorithms.
        :return:
        """
        for agent in self.agents:
            if agent.task or agent.state in [Agent.DESTROYED, Agent.INACTIVE]:
                """Agent already as a task assigned."""

                continue
            self.scheduler.give_task(agent)  # TODO - SJEKK ATTRIBUTE ERROR


