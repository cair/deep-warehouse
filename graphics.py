import pygame
import SharedArray as sa


from pygame import Surface

from deep_logistics import cell_types


class PygameGraphics:

    def __init__(self, environment, game_width, game_height, cell_width, cell_height, has_window=True):
        self.environment = environment
        self.game_width = game_width
        self.game_height = game_height
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.canvas_shape = (self.game_height * self.cell_height, self.game_width * self.cell_width, 3)
        self.changes_cells = []
        self.changes_rects = []
        self.has_window = has_window
        self.rectangles = []
        for x in range(self.game_width):
            for y in range(self.game_height):
                self.rectangles.append(pygame.Rect((x*cell_height, y*cell_width), (cell_width, cell_height)))

        self.environment.grid.cb_on_cell_change.append(self.on_cell_change)

        if self.has_window:
            pygame.display.init()
            self.canvas = pygame.display.set_mode(
                (self.canvas_shape[1], self.canvas_shape[0]),
                0 # TODO NOFRAME OPENGL, HWSURFACE? DOUBLEBUF?
            )
        else:
            self.canvas = Surface((self.canvas_shape[1], self.canvas_shape[0]))

        self.SPRITE_CELL = self._init_sprite(self.bgr2rgb(cell_types.Empty.COLOR), borders=True)
        self.SPRITE_DELIVERY_POINT = self._init_sprite(self.bgr2rgb(cell_types.OrderDelivery.COLOR), borders=True)
        self.SPRITE_DELIVERY_POINT_ACTIVE = self._init_sprite(self.bgr2rgb(cell_types.OrderDeliveryActive.COLOR), borders=True)
        self.SPRITE_PICKUP_POINT = self._init_sprite(self.bgr2rgb(cell_types.OrderPickup.COLOR), borders=True)
        self.SPRITE_SPAWN_POINT = self._init_sprite(self.bgr2rgb(cell_types.SpawnPoint.COLOR), borders=True)
        self.SPRITE_AGENT = self._init_sprite(self.bgr2rgb(cell_types.Agent.COLOR), borders=True)

        self._init_canvas()

    def bgr2rgb(self, bgr):
        return bgr[2], bgr[1], bgr[0]

    def _init_canvas(self):
        """Construct grid."""
        for x in range(self.game_width):
            for y in range(self.game_height):

                cell = self.environment.grid.cell(x, y)

                if cell.type == cell_types.Empty:
                    self.draw_sprite(self.SPRITE_CELL, x, y, setup=True)
                elif cell.type == cell_types.SpawnPoint:
                    self.draw_sprite(self.SPRITE_SPAWN_POINT, x=x, y=y, setup=True)
                elif cell.type == cell_types.OrderDelivery:
                    self.draw_sprite(self.SPRITE_DELIVERY_POINT, x=x, y=y, setup=True)

        if self.has_window:
            pygame.display.update()

    def draw_sprite(self, sprite, x, y, setup=False):
        i = x * self.game_height + y
        self.canvas.blit(sprite, self.rectangles[i])

    def _init_sprite(self, color, borders=False, border_width=1, border_color=(0, 0, 0)):
        rect = pygame.Rect((0, 0, self.cell_width, self.cell_height))
        surf = pygame.Surface((self.cell_width, self.cell_height))

        if borders:
            surf.fill(border_color, rect)
            surf.fill(color, rect.inflate(-border_width*2, -border_width*2))
        else:
            surf.fill(color, rect)

        return surf

    def on_cell_change(self, cell):
        self.changes_cells.append(cell)
        self.changes_rects.append(self.rectangles[cell.i])

    def blit(self):

        for cell in self.changes_cells:
            rect = self.rectangles[cell.i]
            # TODO automate this if clause...
            if cell.occupant:
                self.canvas.blit(self.SPRITE_AGENT, rect)
            elif cell.type == cell_types.Empty:
                self.canvas.blit(self.SPRITE_CELL, rect)
            elif cell.type == cell_types.SpawnPoint:
                self.canvas.blit(self.SPRITE_SPAWN_POINT, rect)
            elif cell.type == cell_types.OrderDelivery:
                self.canvas.blit(self.SPRITE_DELIVERY_POINT, rect)
            elif cell.type == cell_types.OrderDeliveryActive:
                self.canvas.blit(self.SPRITE_DELIVERY_POINT_ACTIVE, rect)
            elif cell.type == cell_types.OrderPickup:
                self.canvas.blit(self.SPRITE_PICKUP_POINT, rect)

        if self.has_window:
            pygame.display.update(self.changes_rects)

        self.changes_rects.clear()
        self.changes_cells.clear()

    def reset(self):
        pass
        #if self.has_window:
        #    for change in self.changes:
        #        pygame.display.update(change)

    def draw_agent(self, agent):
        pass

    def draw_pickup_point(self, x, y):
        pass

    def draw_delivery_point(self, x, y):
        pass

