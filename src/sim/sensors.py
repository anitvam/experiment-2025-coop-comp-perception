import numpy as np

def in_fov(agent_pos, agent_heading, other_pos,
           fov_angle=np.pi/2, max_range=3.0):
    rel = other_pos - agent_pos
    dist = np.linalg.norm(rel)
    if dist > max_range or dist < 1e-6:
        return False

    angle = np.arctan2(rel[1], rel[0])
    delta = np.abs(np.angle(np.exp(1j*(angle - agent_heading))))
    return delta <= fov_angle / 2

def visible_agents(world, agent_id):
    visibles = []
    for j in range(world.n_agents):
        if j == agent_id:
            continue
        if in_fov(
            world.positions[agent_id],
            world.headings[agent_id],
            world.positions[j]
        ):
            visibles.append(j)
    return visibles

def compute_observation(world, agent_id):
    obs = []
    for j in range(world.n_agents):
        if j == agent_id:
            continue
        if in_fov(
            world.positions[agent_id],
            world.headings[agent_id],
            world.positions[j]
        ):
            obs.append(world.positions[j] - world.positions[agent_id])
    if len(obs) == 0:
        obs.append([0.0, 0.0])
    return np.mean(obs, axis=0)
