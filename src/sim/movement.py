import numpy as np

class World:
    def __init__(self, n_agents, world_size=10.0):
        self.n_agents = n_agents
        self.world_size = world_size

        # Initial positioning of agents
        self.positions = np.random.uniform(
            -world_size, world_size, (n_agents, 2)
        )

        self.velocities = np.zeros((n_agents, 2))
        self.headings = np.random.uniform(-np.pi, np.pi, n_agents)
        self.speeds = np.full(n_agents, 1.0)

    def step(self, dt=0.5, turn_std=0.15):
        # heading noise
        dtheta = np.random.normal(
            0.0, turn_std, size=self.n_agents
        )
        self.headings += dtheta

        # update positions
        directions = np.stack(
            [np.cos(self.headings), np.sin(self.headings)],
            axis=1
        )

        self.positions += directions * self.speeds[:, None] * dt

        # reflecting boundaries
        for i in range(self.n_agents):
            for d in range(2):
                if self.positions[i, d] < -self.world_size:
                    self.positions[i, d] = -self.world_size
                    self.headings[i] = np.pi - self.headings[i]
                elif self.positions[i, d] > self.world_size:
                    self.positions[i, d] = self.world_size
                    self.headings[i] = np.pi - self.headings[i]
