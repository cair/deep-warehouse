import ctypes
import random
import SharedArray as sa

from cv2 import FONT_HERSHEY_SIMPLEX, cv2
import numpy as np

import cell_types
from agent import Agent


class Graphics:

    def __init__(self, environment, game_width, game_height, cell_width, cell_height):
        self.environment = environment
        self.game_width = game_width
        self.game_height = game_height
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.canvas_shape = (self.game_height * self.cell_height, self.game_width * self.cell_width, 3)

        """Generate empty canvas template."""
        self._canvas_template = np.ones(shape=self.canvas_shape)
        self.offcreen_canvas = np.array(self._canvas_template, copy=True)

        """Generate shared memory location for canvas"""
        try:
            sa.delete("shm://env_state")
            self.canvas = sa.create("shm://env_state", shape=self.canvas_shape)
        except FileExistsError as e:
            self.canvas = sa.attach("shm://env_state")

        self.SPRITE_CELL = self._plain_cell(borders=True, color=(255, 255, 255))
        self.SPRITE_DELIVERY_POINT = self._plain_cell(borders=True, color=(0, 255, 0))
        self.SPRITE_PICKUP_POINT = self._plain_cell(borders=True, color=(255, 0, 0))
        self.SPRITE_AGENT = {
            Agent.IDLE: self._sprite_agent(cell_types.AgentIdle.COLOR),
            Agent.MOVING_FULL: self._sprite_agent(cell_types.AgentMovingFull.COLOR),
            Agent.MOVING_EMPTY: self._sprite_agent(cell_types.AgentMovingEmpty.COLOR),
            Agent.DIGGING: self._sprite_agent(cell_types.AgentDigging.COLOR)
        }

        self._init_canvas()
        self.reset()

    def _init_canvas(self):
        """Make white."""
        self._canvas_template.fill(255)

        """Construct grid."""
        for y in range(self.game_height):
            for x in range(self.game_width):
                self.draw_sprite(self.SPRITE_CELL, x, y, setup=True)

        for points in [
            self.environment.delivery_points.data,
            self.environment.spawn_points.data
        ]:

            """Setup Cell Color."""
            sprite = np.array(self.SPRITE_CELL, copy=True)
            sprite[1:len(sprite[0]) - 1, 1:len(sprite[1]) - 1] = points[0].type.COLOR

            for point in points:
                """Iterate over all points in the specific group."""
                self.draw_sprite(sprite, x=point.x, y=point.y, setup=True)

    def draw_spawn_point(self, x, y):
        pass

    def draw_delivery_point(self, x, y):
        self.draw_sprite(self.SPRITE_DELIVERY_POINT, x, y)

    def draw_pickup_point(self, x, y):
        self.draw_sprite(self.SPRITE_PICKUP_POINT, x, y)

    def draw_agent(self, agent):
        if agent.state == Agent.INACTIVE:
            return
        sprite = np.array(self.SPRITE_AGENT[agent.state], copy=True)
        # sprite = cv2.putText(sprite, "HI", (0, 0), FONT_HERSHEY_SIMPLEX, 60, cell_types.Colors.BLACK, 2, cv2.LINE_AA)
        self.draw_sprite(sprite, agent.cell.x, agent.cell.y)

    def draw_sprite(self, sprite, x, y, setup=False):
        x_start = x * self.cell_width
        x_end = x_start + self.cell_width
        y_start = y * self.cell_height
        y_end = y_start + self.cell_height

        if setup:
            self._canvas_template[y_start:y_end, x_start:x_end] = sprite
        else:
            self.canvas[y_start:y_end, x_start:x_end] = sprite

    def blit(self):
        pass #np.copyto(self.canvas, self.offcreen_canvas)
        # LETS GO : https://www.pygame.org/docs/tut/newbieguide.html  Update rect based.
    def reset(self):
        np.copyto(self.canvas, self._canvas_template)

    def _plain_cell(self, borders=False, color=(255, 255, 255)):
        """Cell image definition."""
        border_width = 1 if borders else 0
        c = np.ones(shape=(self.cell_height - (border_width * 2), self.cell_width - (border_width * 2), 3))
        for x in range(len(color)):
            c[:, :, x] = color[x]

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

    def _sprite_agent(self, color):
        box = np.zeros(shape=(self.cell_height, self.cell_width, 3))
        box[:, :] = color
        return box
