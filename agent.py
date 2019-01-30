import numpy as np
from action_space import ActionSpace
from grid import Grid


class Agent:
    id = 0

    MAX_SPEED = 750  # X centimeters per second is max speed
    MAX_THRUST = 2

    IDLE = 0
    MOVING_EMPTY = 1
    MOVING_FULL = 2
    DIGGING = 3
    INACTIVE = 4

    @staticmethod
    def new_id():
        _id = Agent.id
        Agent.id += 1
        return _id

    def __init__(self, env):
        self.environment = env
        self.id = Agent.new_id()
        self._cell = None
        self.speed = 0

        self.task = None

        self.state = Agent.INACTIVE  # TODO
        self.victory = False

        self.action = None
        self.action_intensity = 0  # Distance moved in the direction
        self.action_progress = 0  # Accumulator for progress
        self.action_decay_factor = 3

        self.action_steps = {
            ActionSpace.LEFT: 5,  # Number of ticks (Delay) to perform Action.Left
            ActionSpace.RIGHT: 5,
            ActionSpace.UP: 5,
            ActionSpace.DOWN: 5,

        }

    @property
    def cell(self):
        """Ensure consistency between grid and agent."""
        if not self._cell:
            return None
        if not (self._cell.occupant == self or self._cell.occupant is None):
            print(self._cell.occupant, self)
        assert self._cell.occupant == self or self._cell.occupant is None
        return self._cell

    @cell.setter
    def cell(self, x):
        self._cell = x

    def spawn(self, spawn_point):
        try:
            result = self.environment.grid.move(self, spawn_point.x, spawn_point.y)
            assert result == Grid.MOVE_OK
            self.state = Agent.IDLE
        except AssertionError as e:
            """RaceCondition. Another unit moved to spawn tile WHILE async loop ran. TODO"""
            pass

    def despawn(self):
        self.state = Agent.INACTIVE
        self.action = None
        self.action_intensity = 0
        if self.cell:
            self.cell.occupant = None

    def crash(self):
        if self.task:
            self.environment.scheduler.generator.queue.append(self.task)
            self.task.assignee = None
            self.task = None
        self.victory = False
        self.despawn()

    def automate(self):
        return None

    def do_action(self, action):
        if self.action is None or self.action != action:
            self.action = action
        elif action == self.action:
            self.action_intensity = min(Agent.MAX_THRUST, (1 / self.action_steps[action]) + self.action_intensity)

    def update(self):

        if self.state is Agent.INACTIVE:
            return

        if self.action is None:
            return

        action = self.action

        d_prog = ((self.action_intensity * Agent.MAX_SPEED) / Agent.MAX_SPEED) * self.environment.tick_ps_ratio
        self.action_progress += d_prog

        """Calculate number of steps to tage based on the progress"""
        steps = int(self.action_progress)
        self.action_progress -= steps

        assert self.action_progress < 1  # TODO - Remove when release

        x, y = np.multiply(ActionSpace.DIRECTIONS[action], steps)

        return_code = self.environment.grid.move_relative(self, x, y)

        if return_code == Grid.MOVE_WALL_COLLISION:
            print("Wall crash")
            self.crash()
        elif return_code == Grid.MOVE_AGENT_COLLISION:
            # TODO additional handling for other agent
            print("Agent Crash")
            self.crash()

        """Decay acceleration / Thrust."""
        self.action_intensity = max(
            0,
            self.action_intensity - ((1 / (self.action_steps[action] * self.action_decay_factor)) * self.environment.tick_ps_ratio)
        )


class ManhattanAgent(Agent):

    def __init__(self, env):
        super().__init__(env)

    def automate(self):
        if self.task:
            # +dY = Above
            # -dY = Below
            # +dX = Right Of
            # -dX = Left Of
            task_coords = self.task.get_coordinates()

            d_x = self.cell.x - task_coords.x
            d_y = self.cell.y - task_coords.y

            is_aligned_x = d_x == 0
            is_aligned_y = d_y == 0

            if not is_aligned_x:
                if d_x > 0:
                    self.do_action(ActionSpace.LEFT)
                else:
                    self.do_action(ActionSpace.RIGHT)
            elif not is_aligned_y:
                if d_y > 0:
                    self.do_action(ActionSpace.UP)
                else:
                    self.do_action(ActionSpace.DOWN)

            #print("x=%s | y=%s | dX=%s | dY=%s | Thrust=%s | alignment_x=%s | alignment_y=%s" %
            #      (self.cell.x, self.cell.y, d_x, d_y, self.action_intensity, is_aligned_x, is_aligned_y))
