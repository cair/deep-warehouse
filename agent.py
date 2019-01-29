import random

from action_space import ActionSpace
from cell import Cell
from grid import Grid


class Agent:
    id = 0

    MAX_SPEED = 750  # X centimeters per second is max speed

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
        self.cell = None
        self.speed = 0

        self.task = None

        self.state = Agent.INACTIVE  # TODO
        self.victory = False

        self.action = None
        self.action_intensity = 0  # Distance moved in the direction
        self.action_progress = 0  # Accumulator for progress
        self.action_decay_factor = 0

        self.action_steps = {
            ActionSpace.LEFT: 5,  # Number of ticks (Delay) to perform Action.Left
            ActionSpace.RIGHT: 5,
            ActionSpace.UP: 5,
            ActionSpace.DOWN: 5,

        }

    def spawn(self, spawn_point):
        spawn_point.occupant = self
        self.cell = spawn_point

        self.state = Agent.IDLE

    def despawn(self):
        self.state = Agent.INACTIVE
        self.action = None
        self.action_intensity = 0

    async def crash(self):
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
            self.action_intensity = min(1, (1 / self.action_steps[action]) + self.action_intensity)

    def update(self):

        if self.state is Agent.INACTIVE:
            return

        if self.action is not None:

            """Update pixel value"""
            moved_centimeters = (self.action_intensity * Agent.MAX_SPEED)
            self.action_progress += (moved_centimeters / Cell.CELL_DIMENSION)

            if self.action_progress >= 1:
                self.action_progress -= 1  # TODO, may break in some cases?

                """Determine which direction the agent should move."""
                x, y = ActionSpace.DIRECTIONS[self.action]


                return_code = self.environment.grid.move_relative(self, x, y)

                if return_code == Grid.MOVE_WALL_COLLISION:
                    self.crash()
                    return

                elif return_code == Grid.MOVE_AGENT_COLLISION:
                    # TODO additional handling for other agent
                    self.crash()
                    return

        """Evaluate task objective."""
        if self.task:
            if self.task.at_location():
                self.task.signal()


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
