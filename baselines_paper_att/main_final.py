from flow.core.params import TrafficLightParams
from flow.envs.ring.accel import AccelEnv
from flow.core.params import InitialConfig
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import IDMController
from flow.core.params import NetParams
from flow.utils.registry import make_create_env
import numpy as np
import importlib
networks = []
class_name = 'IntersectionNetwork'
for i in range(81):
    scenario_module = importlib.import_module('baselines_paper_att.scenarios.scenario' + str(i))
    networks.append(getattr(scenario_module, class_name))
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.core.params import EnvParams
from flow.core.params import SumoParams
from flow.core.params import InFlows

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()

traffic_lights = TrafficLightParams()
sim_params = SumoParams(sim_step=0.1, render=False, seed=args.seed)
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
initial_config = InitialConfig()
vehicles = VehicleParams()
vehicles.add("human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=0,
             car_following_params=SumoCarFollowingParams(speed_mode='right_of_way'),
             )

inflow = InFlows()
inflow.add(veh_type="human",
           edge="edge-north-NS",
           depart_speed="max",
           vehs_per_hour=1,
           #period=1200,
           #end=5.0
          )
inflow.add(veh_type="human",
           edge="edge-south-SN",
           depart_speed="max",
           vehs_per_hour=1,
           #period=1200,
           #end=5.0
          )
inflow.add(veh_type="human",
           edge="edge-east-EW",
           depart_speed="max",
           vehs_per_hour=1,
           #period=1200,
           #end=5.0
          )
inflow.add(veh_type="human",
           edge="edge-west-WE",
           depart_speed="max",
           vehs_per_hour=1,
           #period=1200,
           #end=5.0
          )
# net_params = NetParams(inflows=inflow, template='map.net.xml')
net_params = NetParams(inflows=inflow, template='map.net.xml')

fp_list = []

for i in range(81):
    flow_params = dict(
        exp_tag='baseline',
        env_name=AccelEnv,
        network=networks[i],
        simulator='traci',
        sim=sim_params,
        env=env_params,
        net=net_params,
        veh=vehicles,
        initial=initial_config,
        tls=traffic_lights,
    )
    # number of time steps
    flow_params['env'].horizon = 1200
    fp_list.append(flow_params)


def rl_actions(*_):
    return None


num_runs = 81
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

np.save(f'travel_time_fttlW_{args.seed}.npy', travel_time)
np.save(f'waiting_time_fttlW_{args.seed}.npy', waiting_time)
np.save(f'avg_speed_fttlW_{args.seed}.npy', avg_speed)

print(f'Finished seed {args.seed}')
