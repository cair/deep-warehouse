import numpy as np
from action_space import ActionSpace
from grid import Grid


class Agent:
    id = 0

    MAX_SPEED = 750  # X centimeters per second is max speed
    MAX_THRUST = 2

    IDLE = 0
    MOVING = 1
    PICKUP = 2
    DELIVERY = 3
    DESTROYED = 4
    INACTIVE = 5

    ALL_STATES = [IDLE, MOVING, PICKUP, DELIVERY, DESTROYED, INACTIVE]

    IMMOBILE_STATES = [DESTROYED, INACTIVE]

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

        assert self._cell.occupant == self or self._cell.occupant is None
        return self._cell

    @cell.setter
    def cell(self, x):

        _cell = self.cell
        if _cell:
            _cell.occupant = None

        self._cell = x

    def spawn(self, spawn_point):
        result = self.environment.grid.move(self, spawn_point.x, spawn_point.y)
        assert result == Grid.MOVE_OK
        self.state = Agent.IDLE

    def despawn(self):
        self.reset_action()
        self.cell = None
        self.state = Agent.INACTIVE

    def crash(self):

        if self.task:
            self.task.abort()
            self.task = None

        self.state = Agent.DESTROYED
        self.reset_action()
        self.cell = None

    def reset_action(self):
        self.action = None
        self.action_intensity = 0

    def automate(self):
        return None

    def do_action(self, action):
        if self.state in Agent.IMMOBILE_STATES:
            return

        if action < 0 or action >= ActionSpace.N_ACTIONS:
            raise ValueError("The inserted action is out of action_space bounds 0 => %s." % ActionSpace.N_ACTIONS)

        """Ensure that action is Integer"""
        action = int(action)

        if action is ActionSpace.NOOP:
            return
        elif self.action is None or self.action != action:
            self.action = action
        elif action == self.action:
            self.action_intensity = min(Agent.MAX_THRUST, (1 / self.action_steps[action]) + self.action_intensity)

    def update(self):

        if self.state is Agent.INACTIVE:
            return
        elif self.state is Agent.DESTROYED:
            self.state = Agent.INACTIVE

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
        self.state = Agent.MOVING

        if return_code == Grid.MOVE_WALL_COLLISION:
            #print("Wall crash")
            self.crash()
            return
        elif return_code == Grid.MOVE_AGENT_COLLISION:
            # TODO additional handling for other agent
            #print("Agent Crash")
            self.crash()
            return

        """Decay acceleration / Thrust."""
        self.action_intensity = max(
            0,
            self.action_intensity - ((1 / (self.action_steps[action] * self.action_decay_factor)) * self.environment.tick_ps_ratio)
        )

        if self.action_intensity == 0:
            self.state = Agent.IDLE


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
