import numpy as np

from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, VehicleParams, InFlows
from flow.utils.registry import make_create_env
from intersection_network import IntersectionNetwork, ADDITIONAL_NET_PARAMS
from intersection_env import MyEnv, ADDITIONAL_ENV_PARAMS
from utils import compute_rp

vehicles = VehicleParams()
vehicles.add(veh_id="rl",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=4,
             color='green')
# inflow = InFlows()
# inflow.add(veh_type="rl",
#            edge="b_c",
#            vehs_per_hour="1"
#            #probability=0.05,
#            #depart_speed="random",
#           )
# inflow.add(veh_type="rl",
#            edge="t_c",
#            probability=0.1,
#            #depart_speed="random",
#           )
# inflow.add(veh_type="rl",
#            edge="l_c",
#            probability=0.1,
#            #depart_speed="random",
#           )
# inflow.add(veh_type="rl",
#            edge="r_c",
#            probability=0.05,
#            #depart_speed="random",
#           )
sim_params = SumoParams(sim_step=0.1, render=False)
initial_config = InitialConfig()
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
additional_net_params = ADDITIONAL_NET_PARAMS.copy()
net_params = NetParams(additional_params=additional_net_params)

flow_params = dict(
    exp_tag='test_network',
    env_name=MyEnv,
    network=IntersectionNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

flow_params['env'].horizon = 1000
create_env, _ = make_create_env(flow_params)
env = create_env()

num_steps = env.env_params.horizon
eval_returns = []
steps_per_ep = []

rl_actions = []
for i in range(10):
    ep_steps = 0
    ret = 0
    state = env.reset()

    for j in range(num_steps):

        # HERE ACTION SELECTION STEP
        state_, reward, done, _ = env.step(rl_actions=rl_actions)
        reward = compute_rp(state, reward)

        ep_steps += 1
        state = state_
        ret += reward
        if done or state_.pos is None:
            steps_per_ep.append(ep_steps)
            eval_returns.append(ret)
            break

env.terminate()
print(f'Average return: {np.mean(eval_returns)}, Average ep steps: {np.mean(steps_per_ep)}')
