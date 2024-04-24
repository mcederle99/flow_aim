from flow.core.params import TrafficLightParams
from flow.envs.ring.accel import AccelEnv
from flow.core.params import InitialConfig
from flow.core.params import VehicleParams
from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import IDMController
from flow.core.params import NetParams
import importlib
networks = []
class_name = 'IntersectionNetwork'
for i in range(1):
    scenario_module = importlib.import_module('baselines_paper.scenarios.scenario' + str(i))
    networks.append(getattr(scenario_module, class_name))
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.core.params import EnvParams
from flow.core.params import SumoParams
from flow.core.params import InFlows

traffic_lights = TrafficLightParams()
sim_params = SumoParams(sim_step=0.1, render=True)
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
initial_config = InitialConfig()
vehicles = VehicleParams()
vehicles.add("human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=0,
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
net_params = NetParams(inflows=inflow, template='map.net.xml')

fp_list = []

for i in range(1):
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