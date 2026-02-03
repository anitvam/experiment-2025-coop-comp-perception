import numpy as np

def in_fov(agent_pos, agent_heading, other_pos, fov_angle=np.pi/2, max_range=3.0):
    rel = other_pos - agent_pos
    dist = np.linalg.norm(rel)
    if dist > max_range or dist < 1e-6:
        return False

    angle = np.arctan2(rel[1], rel[0])
    delta = np.abs(np.angle(np.exp(1j*(angle - agent_heading))))
    return delta <= fov_angle / 2

def visible_agents(world, agent_id, radius=4.0, fov=np.pi / 2):
    pos_i = world.positions[agent_id]
    vel_i = world.velocities[agent_id]

    speed = np.linalg.norm(vel_i)

    # if agent is almost stationary, it sees nothing (or everything, see note below)
    if speed < 1e-5:
        return []

    # heading unit vector
    h = vel_i / speed

    visible = []

    for j in range(world.n_agents):
        if j == agent_id:
            continue

        delta = world.positions[j] - pos_i
        dist = np.linalg.norm(delta)

        # distance check
        if dist > radius or dist < 1e-6:
            continue

        # angle check via dot product
        d_hat = delta / dist
        cos_angle = np.dot(h, d_hat)

        # clamp for numerical safety
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle = np.arccos(cos_angle)

        if angle <= fov / 2:
            visible.append(j)

    return visible

def compute_observation(world, agent_id):
    obs = []
    for j in range(world.n_agents):
        if j == agent_id:
            continue
        if in_fov(
            world.positions[agent_id],
            world.velocities[agent_id],
            world.positions[j]
        ):
            obs.append(world.positions[j] - world.positions[agent_id])
    if len(obs) == 0:
        obs.append([0.0, 0.0])
    return np.mean(obs, axis=0)
