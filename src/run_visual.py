import time
from env.custom_environment import CustomEnvironment

env = CustomEnvironment(n_agents=6)
obs, _ = env.reset()

for _ in range(1000):
    obs, _, _, _, _ = env.step()
    env.render()
    time.sleep(0.3)
