import pygame
import numpy as np

class Renderer:
    def __init__(self, world, scale=40):
        pygame.init()
        self.world = world
        self.scale = scale
        self.size = 600
        self.screen = pygame.display.set_mode(
            (self.size, self.size)
        )
        self.center = self.size // 2

    def world_to_screen(self, pos):
        return (
            int(self.center + pos[0] * self.scale),
            int(self.center - pos[1] * self.scale),
        )

    def draw_agent(self, i):
        pos = self.world.positions[i]
        heading = self.world.headings[i]

        p = self.world_to_screen(pos)
        pygame.draw.circle(self.screen, (0, 0, 255), p, 6)

        # heading line
        end = (
            pos[0] + 0.7 * np.cos(heading),
            pos[1] + 0.7 * np.sin(heading),
        )
        pygame.draw.line(
            self.screen, (0, 0, 0),
            p, self.world_to_screen(end), 2
        )

        # FOV cone
        fov = np.pi / 2
        for a in np.linspace(-fov / 2, fov / 2, 15):
            ray = (
                pos[0] + 4 * np.cos(heading + a),
                pos[1] + 4 * np.sin(heading + a),
            )
            pygame.draw.line(
                self.screen, (200, 200, 200),
                p, self.world_to_screen(ray), 1
            )

    def render(self):
        self.screen.fill((255, 255, 255))
        for i in range(self.world.n_agents):
            self.draw_agent(i)
        pygame.display.flip()
