from flow.utils.registry import make_create_env
import numpy as np
from utils import fp_list


def rl_actions(*_):
    return None


num_runs = 1
num_steps = 1200

collisions = []
travel_time = []
waiting_time = []
avg_speed = []

for i in range(num_runs):
    vel = []
    tts = [0, 0, 0, 0]
    ttl = [0, 0, 0, 0]
    done, crash = False, False
    create_env, _ = make_create_env(fp_list[i])
    env = create_env()
    _ = env.reset()

    for j in range(num_steps):
        _, _, crash, _ = env.step(rl_actions())

        veh_ids = env.k.vehicle.get_ids()
        speeds = []
        for idx, vid in enumerate(veh_ids):
            tts[idx] += env.k.vehicle.get_timedelta(vid)
            speed = env.k.vehicle.get_speed(vid)
            speeds.append(speed)
            if speed < 0.1:
                ttl[idx] += env.k.vehicle.get_timedelta(vid)

        num_vehs = len(list(veh_ids))
        if j > 10:
            if num_vehs != 0:
                vel.append(np.mean(speeds))
            else:
                done = True
        if crash or done:
            break

    travel_time.append(np.mean(tts))
    avg_speed.append(np.mean(vel))
    waiting_time.append(np.mean(ttl))
    if crash:
        collisions.append(1)
    else:
        collisions.append(0)

    env.terminate()

print(travel_time)
print(avg_speed)
print(waiting_time)
print(collisions)
