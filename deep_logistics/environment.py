import time
import numpy as np
from deep_logistics.action_space import ActionSpace
from deep_logistics.agent import ManhattanAgent, Agent
from deep_logistics.delivery_points import DeliveryPointGenerator
from deep_logistics.graphics import PygameGraphics
from deep_logistics.grid import Grid
from deep_logistics.scheduler import OnDemandScheduler

from deep_logistics.agent_storage import AgentStore
from deep_logistics.spawn_strategy import RandomSpawnStrategy, LocationSpawnStrategy


class Environment:
    """
    Environment Class, Remember that Numpy Operates with arr[y, x]
    """

    def __init__(self,
                 height,
                 width,
                 depth,
                 ups=None,
                 ticks_per_second=10,
                 taxi_n=1,
                 taxi_agent=ManhattanAgent,
                 taxi_respawn=False,  # TODO - There is no case where this will happen in reality
                 scheduler=OnDemandScheduler,
                 delivery_locations=None,
                 spawn_strategy=LocationSpawnStrategy,
                 graphics_render=False,
                 graphics_tile_width=32,
                 graphics_tile_height=32
                 ):
        super().__init__()

        self.width = width
        self.height = height
        self.depth = depth

        self.taxi_respawn = taxi_respawn

        """Updates per second."""
        self.ups = ups
        self.ups_interval = 0 if self.ups is None else 1.0 / self.ups

        """Ticks per second."""
        self.tick_ps = ticks_per_second
        self.tick_ps_ratio = 1 / self.tick_ps
        self.tick_ps_counter = 0

        """The grid is the global internal state of all cells in the environment."""
        self.grid = Grid(width=width, height=height)

        """Spawn-points is the location where agent can spawn."""
        self.spawn_points = spawn_strategy(self, seed=12)

        """Delivery points is a (TODO) static definition for where agents can deliver scheduled tasks."""
        self.delivery_points = DeliveryPointGenerator(self, override=delivery_locations, seed=555)

        """The scheduler is a engine for scheduling tasks to agents."""
        self.scheduler = scheduler(self)

        """List of all available agents."""
        if taxi_n < 1:
            raise ValueError("There must be AT LEAST one initial agent!.")
        self.agents = AgentStore(self)
        self.agents.add_agent(
            cls=taxi_agent,
            n=taxi_n
        )
        self.selected_agent = self.agents[0]

        """Action Space + Observation space"""
        self.action_space = ActionSpace

        """GUIComponents is a subclass used for rending the internal state of the environment."""
        self.graphics = PygameGraphics(environment=self,
                                       game_width=self.width,
                                       game_height=self.height,
                                       cell_width=graphics_tile_width,
                                       cell_height=graphics_tile_height,
                                       has_window=graphics_render
                                       )

        """Reset environment."""
        self.reset()

    def get_agent(self, idx) -> Agent:
        return self.agents[idx]

    def get_seconds(self):
        return self.tick_ps_counter * self.tick_ps_ratio

    def is_terminal(self):
        """Check if game is terminal TODO - This yields true if ANY of the agents has crashed."""
        for a in self.agents:
            if a.is_terminal():
                return True
        return False

    def update(self):
        self.tick_ps_counter += 1

        """Process agent s."""
        for agent in self.agents:

            agent.automate()
            agent.update()

            """Evaluate task objective."""
            if agent.task:
                agent.task.evaluate()
            else:
                agent.request_task()

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
                raise RuntimeWarning("There is no available spawn points!")
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
            self.scheduler.give_task(agent)

    def reset(self):
        for agent in self.agents:
            agent.despawn()
        self.deploy_agents()
        self.task_assignment()
