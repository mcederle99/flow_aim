import numpy as np

from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.networks.base import Network

ADDITIONAL_NET_PARAMS = {
    # radius of the intersection
    "radius_intersection": 15,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 13.9,
    # resolution of the curved portions
    "resolution": 40
}

class IntersectionNetwork(Network):
    """ Requires from net_params:

    * **radius_intersection** : radius of the intersection
    * **resolution** : number of nodes resolution in the circular portions
    * **lanes** : number of lanes in the network
    * **speed** : max speed of vehicles in the network

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from road_network import IntersectionNetwork
    >>>
    >>> network = IntersectionNetwork(
    >>>     name='intersection',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'radius_intersection': 15,
    >>>             'lanes': 3,
    >>>             'speed_limit': 13.9,
    >>>             'resolution': 40
    >>>         },
    >>>     )
    >>> )
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):

        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        self.intersection_len = 100

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        r = net_params.additional_params["radius_intersection"]

        nodes = [{
            "id": "center",
            "x": 0,
            "y": 0,
            "radius": r,
            "type": "priority"
        }, {
            "id": "right",
            "x": self.intersection_len,
            "y": 0,
            "type": "priority"
        }, {
            "id": "top",
            "x": 0,
            "y": self.intersection_len,
            "type": "priority"
        }, {
            "id": "left",
            "x": -self.intersection_len,
            "y": 0,
            "type": "priority"
        }, {
            "id": "bottom",
            "x": 0,
            "y": -self.intersection_len,
            "type": "priority"
        }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""

        # intersection edges
        edges = [{
            "id": "b_c",
            "type": "edgeType",
            #"priority": "78",
            "from": "bottom",
            "to": "center",
            "length": self.intersection_len
        }, {
            "id": "c_t",
            "type": "edgeType",
            #"priority": 78,
            "from": "center",
            "to": "top",
            "length": self.intersection_len
        }, {
            "id": "r_c",
            "type": "edgeType",
            #"priority": 78,
            "from": "right",
            "to": "center",
            "length": self.intersection_len
        }, {
            "id": "c_l",
            "type": "edgeType",
            #"priority": 46,
            "from": "center",
            "to": "left",
            "length": self.intersection_len
        }, {
            "id": "t_c",
            "type": "edgeType",
            #"priority": 78,
            "from": "top",
            "to": "center",
            "length": self.intersection_len
        }, {
            "id": "c_r",
            "type": "edgeType",
            #"priority": 46,
            "from": "center",
            "to": "right",
            "length": self.intersection_len
        }, {
            "id": "l_c",
            "type": "edgeType",
            #"priority": 78,
            "from": "left",
            "to": "center",
            "length": self.intersection_len
        }, {
            "id": "c_b",
            "type": "edgeType",
            #"priority": "78",
            "from": "center",
            "to": "bottom",
            "length": self.intersection_len
        }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        lanes = net_params.additional_params["lanes"]
        speed_limit = net_params.additional_params["speed_limit"]
        types = [{
            "id": "edgeType",
            "numLanes": lanes,
            "speed": speed_limit
        }]

        return types
    
    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "r_c":
                [(["r_c", "c_l"], 1/3), (["r_c", "c_t"], 1/3),
                    (["r_c", "c_b"], 1/3)],
            "b_c":
                [(["b_c", "c_l"], 1/3), (["b_c", "c_t"], 1/3),
                    (["b_c", "c_r"], 1/3)],
            "t_c":
                [(["t_c", "c_b"], 1/3), (["t_c", "c_l"], 1/3),
                    (["t_c", "c_r"], 1/3)],
            "l_c":
                [(["l_c", "c_r"], 1/3), (["l_c", "c_t"], 1/3),
                    (["l_c", "c_b"], 1/3)],
            "c_r":
                ["c_r"],
            "c_l":
                ["c_l"],
            "c_t":
                ["c_t"],
            "c_b":
                ["c_b"],
            "rl_0":
                ["r_c", "c_l"],
            "rl_1":
                ["t_c", "c_b"],
            "rl_2":
                ["l_c", "c_t"]
            }

        return rts
