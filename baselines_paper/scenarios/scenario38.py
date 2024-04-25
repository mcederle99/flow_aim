from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.networks.base import Network
from scenarios.routes import routes_list

class IntersectionNetwork(Network):
    """ Requires from net_params:

    * **radius_intersection** : radius of the intersection
    * **resolution** : number of nodes resolution in the circular portions
    * **lanes** : number of lanes in the network
    * **speed** : max speed of vehicles in the network
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "edge-east-EW": ["edge-east-EW", routes_list[38][0]],
                # [(["edge-east-EW", "edge-west-EW"], 1/3), (["edge-east-EW", "edge-north-SN"], 1/3),
                #     (["edge-east-EW", "edge-south-NS"], 1/3)],
            "edge-south-SN": ["edge-south-SN", routes_list[38][1]],
                # [(["edge-south-SN", "edge-west-EW"], 1/3), (["edge-south-SN", "edge-north-SN"], 1/3),
                #     (["edge-south-SN", "edge-east-WE"], 1/3)],
            "edge-north-NS": ["edge-north-NS", routes_list[38][2]],
                # [(["edge-north-NS", "edge-south-NS"], 1/3), (["edge-north-NS", "edge-west-EW"], 1/3),
                #     (["edge-north-NS", "edge-east-WE"], 1/3)],
            "edge-west-WE": ["edge-west-WE", routes_list[38][3]],
                # [(["edge-west-WE", "edge-east-WE"], 1/3), (["edge-west-WE", "edge-north-SN"], 1/3),
                #     (["edge-west-WE", "edge-south-NS"], 1/3)],
            "edge-east-WE":
                ["edge-east-WE"],
            "edge-west-EW":
                ["edge-west-EW"],
            "edge-north-SN":
                ["edge-north-SN"],
            "edge-south-NS":
                ["edge-south-NS"],
            # "human_0":
            #     ["edge-north-NS", "edge-south-NS"],
            # "human_1":
            #     ["edge-south-SN", "edge-north-SN"],
            # "human_2":
            #     ["edge-east-EW", "edge-west-EW"],
            # "human_3":
            #     ["edge-west-WE", "edge-east-WE"],
            }

        return rts
