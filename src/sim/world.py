import numpy as np

class World:
    def __init__(self, n_agents, world_size=10.0):
        self.n_agents = n_agents
        self.world_size = world_size
        self.positions = np.random.uniform(
            -world_size, world_size, (n_agents, 2)
        )
        self.headings = np.random.uniform(-np.pi, np.pi, n_agents)

    def step(self, actions):
        # actions: [dx, dy]
        self.positions += actions
        self.positions = np.clip(
            self.positions, -self.world_size, self.world_size
        )
