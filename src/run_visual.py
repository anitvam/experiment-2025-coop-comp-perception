import time
import numpy as np
from env.custom_environment import CustomEnvironment

env = CustomEnvironment(n_agents=6)
obs, _ = env.reset()

for _ in range(1000):
    actions = {
        a: np.random.uniform(-0.05, 0.05, 2)
        for a in env.agents
    }
    obs, _, _, _, _ = env.step(actions)
    env.render()
    time.sleep(0.03)
