import asyncio
import numpy as np
import cv2

import cell_types
from action_space import ActionSpace
from agent import Agent
from delivery_points import DeliveryPointGenerator
from scheduler import RandomScheduler
from spawn_points import SpawnPoints


class Cell:
    CELL_DIMENSION = 1000  # X centimeters in width

    def __init__(self, y, x):
        self.type = cell_types.EmptyPoint
        self.occupant = None
        self.x = x
        self.y = y


class Environment:
    class GUIComponents:

        def __init__(self, environment, depth=5, width=32, height=32):
            self.width = width
            self.height = height
            self.depth = depth
            self.gw = environment.w
            self.gh = environment.h
            self.environment = environment

            self.font = cv2.FONT_HERSHEY_SIMPLEX
            self.cell = self.define_cell()
            self.agent = {
                Agent.IDLE: self.define_agent(cell_types.AgentIdle.COLOR),
                Agent.MOVING_FULL: self.define_agent(cell_types.AgentMovingFull.COLOR),
                Agent.MOVING_EMPTY: self.define_agent(cell_types.AgentMovingEmpty.COLOR),
                Agent.DIGGING: self.define_agent(cell_types.AgentDigging.COLOR)
            }

            self._canvas = self.init_canvas()

        def define_agent(self, color):
            box = np.zeros(shape=(self.height, self.width, 3))
            box[:, :] = color
            return box

        def init_canvas(self):
            empty_grid = np.ones(shape=(self.gh * self.height, self.gw * self.width, 3))
            empty_grid.fill(255)

            """ Draw Grid"""
            for gy in range(self.gh):
                for gx in range(self.gw):
                    """Convert gy, and gx to pixels in the canvas."""
                    x_start = gx * self.width
                    x_end = x_start + self.width
                    y_start = gy * self.height
                    y_end = y_start + self.height

                    empty_grid[y_start:y_end, x_start:x_end] = self.cell

            for points in [self.environment.delivery_points.data, self.environment.spawn_points.data]:

                """Setup Cell Color."""
                cell = np.array(self.cell, copy=True)
                cell[1:len(cell[0])-1, 1:len(cell[1])-1] = points[0].type.COLOR  # TODO may be heavy

                for point in points:
                    """Iterate over all points in the specific group."""

                    y_start = point.y * self.height
                    y_end = y_start + self.height
                    x_start = point.x * self.width
                    x_end = x_start + self.width

                    empty_grid[y_start:y_end, x_start:x_end] = cell

            return empty_grid

        def get_canvas(self):
            return np.array(self._canvas, copy=True)

        def generate_agent(self, agent):
            _agent = np.array(self.agent[agent.state], copy=True)
            cv2.putText(_agent, str(agent.id), (0, 0), self.font, 6, cell_types.Colors.BLACK, 2, cv2.LINE_AA)
            return _agent



        def define_cell(self):
            """Cell image definition."""
            border_width = 1
            c = np.ones(shape=(self.height - (border_width * 2), self.width - (border_width * 2), 3))
            c *= 255
            c = cv2.copyMakeBorder(
                c,
                border_width,
                border_width,
                border_width,
                border_width,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)  # Black border
            )
            return c

    def __init__(self, loop, height, width, depth, agents=1, agent_class=Agent, renderer=None, tile_height=32, tile_width=32, scheduler=RandomScheduler):
        """Class references to prevent cyclic imports."""
        self.Cell = Cell

        self.w = width
        self.h = height
        self.d = depth
        self.loop = loop
        self.action_space = ActionSpace
        self.renderer = renderer

        """The grid is the global internal state of all cells in the environment."""
        self.grid = self.init_grid()

        """Spawn-points is the location where agent can spawn."""
        self.spawn_points = SpawnPoints(loop, self, seed=123)

        """Delivery points is a (TODO) static definition for where agents can deliver scheduled tasks."""
        self.delivery_points = DeliveryPointGenerator(loop, self, seed=123)

        """The scheduler is a engine for scheduling tasks to agents."""
        self.scheduler = scheduler(loop, self)

        """List of all available agents."""
        self.agents = [agent_class(loop, self) for _ in range(agents)]

        """Current selected agent."""
        self.agent = None

        """GUIComponents is a subclass used for rending the internal state of the environment."""
        self.gui_components = Environment.GUIComponents(self, height=tile_height, width=tile_width)

        """Validate the setup."""
        self.ensure_grid_consistency()

        """Setup tasks."""
        self.loop.create_task(self.deploy_agents())
        self.loop.create_task(self.scheduler.generator.generate())
        self.loop.create_task(self.task_assignment())

    def add_agent(self, agent_cls):
        idx = len(self.agents)
        self.agents.append(agent_cls(self.loop, self))
        return self.agents[idx]

    def init_grid(self):
        """Create grid data information"""
        grid = np.ndarray(shape=(self.h, self.w), dtype=np.object)
        for (a, b), index in np.ndenumerate(grid):
            grid[a, b] = Cell(y=a, x=b)

        return grid

    def ensure_grid_consistency(self):
        """
        Ensures that the grid has a type set for all cells.
        :return:
        """

        n_counts = {
            cell_types.DeliveryPoint: len(self.delivery_points.data),
            cell_types.SpawnPoint: len(self.spawn_points.data),
            cell_types.EmptyPoint: (self.h * self.w) - len(self.delivery_points.data) - len(self.spawn_points.data)
        }

        counts = {
            cell_types.DeliveryPoint: 0,
            cell_types.SpawnPoint: 0,
            cell_types.EmptyPoint: 0
        }

        for (a, b), index in np.ndenumerate(self.grid):
            cell = self.grid[a, b]
            assert cell.type

            counts[cell.type] += 1

        #for k, v in n_counts.items():
        #    assert v == counts[k] # TODO reimplement

    def set_agent(self, agent):
        """
        Sets a specific agent to be main environment pov.
        :param agent:
        :return:
        """
        self.agent = agent

    async def step(self, action):
        await self.agent.do_action(action=action)

        await asyncio.sleep(.01)

    async def update(self):
        for agent in self.agents:
            await agent.update()

    async def deploy_agents(self):
        """
        Deploy agent if there are any in the queue.
        :return:
        """
        while True:
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

            await asyncio.sleep(.5)

    async def preprocess(self):
        _canvas = self.gui_components.get_canvas()
        for agent in self.agents:
            if not agent.x or not agent.y:
                continue

            if agent.task:
                """Draw task location."""
                order_x_start = agent.task.order_x * self.gui_components.width
                order_x_end = order_x_start + self.gui_components.width
                order_y_start = agent.task.order_y * self.gui_components.height
                order_y_end = order_y_start + self.gui_components.height

                delivery_x_start = agent.task.delivery_x * self.gui_components.width
                delivery_x_end = delivery_x_start + self.gui_components.width
                delivery_y_start = agent.task.delivery_y * self.gui_components.height
                delivery_y_end = delivery_y_start + self.gui_components.height

                if not agent.task.picked_up:
                    _canvas[order_y_start:order_y_end, order_x_start:order_x_end] = cell_types.Order.COLOR_0
                if not agent.task.done:
                    _canvas[delivery_y_start:delivery_y_end, delivery_x_start:delivery_x_end] = cell_types.Colors.BLACK


            x = agent.x
            y = agent.y
            img_x_start = x * self.gui_components.width
            img_x_end = img_x_start + self.gui_components.width
            img_y_start = y * self.gui_components.height
            img_y_end = img_y_start + self.gui_components.height

            _canvas[img_y_start:img_y_end, img_x_start:img_x_end] = self.gui_components.generate_agent(agent)


        return _canvas
