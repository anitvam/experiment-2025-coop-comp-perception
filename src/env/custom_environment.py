from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np

from sim.movement import World
from sim.sensors import visible_agents
from sim.renderer import Renderer

class CustomEnvironment(ParallelEnv):
    """The ParallelEnv class steps every live agent at once."""

    # metadata holds environment constants
    metadata = {
        "name": "custom_environment",
        "render_modes": ["human"]
    }

    def __init__(self, n_agents=5):
        self.n_agents = n_agents
        self.agents = [f"agent_{i}" for i in range(n_agents)]

        self.world = World(n_agents)
        self.renderer = Renderer(self.world)

    def reset(self, seed=None, options=None):
        """Resets the environment to a starting point. Returns a dictionary of observations"""
        self.world = World(self.n_agents)
        self.renderer.world = self.world
        obs = self._observe()
        return obs, {}

    def _observe(self):
        obs = {}
        for i, a in enumerate(self.agents):
            visible = visible_agents(self.world, i)
            mask = np.zeros(self.n_agents)
            mask[visible] = 1.0
            obs[a] = mask
        return obs

    def step(self, dt=0.1, turn_std=0.15):
        self.world.step(dt, turn_std)

        obs = self._observe()
        rewards = {a: 0.0 for a in self.agents}
        terms = {a: False for a in self.agents}
        truncs = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        return obs, rewards, terms, truncs, infos

    def render(self):
        self.renderer.render()
