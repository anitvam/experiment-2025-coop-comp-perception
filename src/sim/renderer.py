import pygame
import numpy as np

class Renderer:
    def __init__(self, world, scale=30):
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
        radius = 4.0
        color = (200, 200, 255, 80)

        surface = pygame.Surface(
            (self.size, self.size), pygame.SRCALPHA
        )

        points = [self.world_to_screen(pos)]

        for angle in np.linspace(-fov / 2, fov / 2, 30):
            ray = (
                pos[0] + radius * np.cos(heading + angle),
                pos[1] + radius * np.sin(heading + angle),
            )
            points.append(self.world_to_screen(ray))

        pygame.draw.polygon(surface, color, points)
        self.screen.blit(surface, (0, 0))

    def render(self):
        self.screen.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        for i in range(self.world.n_agents):
            self.draw_agent(i)
        pygame.display.flip()
