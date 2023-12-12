import numpy as np
from numpy import pi, sin, cos, linspace

from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.networks.base import Network

ADDITIONAL_NET_PARAMS = {
    # radius of the circular components
    "radius_ring": 30,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # resolution of the curved portions
    "resolution": 40
}

class IntersectionNetwork(Network):
    """Figure eight network class.

    The figure eight network is an extension of the ring road network: Two
    rings, placed at opposite ends of the network, are connected by an
    intersection with road segments of length equal to the diameter of the
    rings. Serves as a simulation of a closed ring intersection.

    Requires from net_params:

    * **ring_radius** : radius of the circular portions of the network. Also
      corresponds to half the length of the perpendicular straight lanes.
    * **resolution** : number of nodes resolution in the circular portions
    * **lanes** : number of lanes in the network
    * **speed** : max speed of vehicles in the network

    Usage
    -----
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize a figure 8 network."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        ring_radius = net_params.additional_params["radius_ring"]
        self.ring_edgelen = ring_radius * np.pi / 2.
        self.intersection_len = 2 * ring_radius
        self.junction_len = 2.9 + 3.3 * net_params.additional_params["lanes"]
        self.inner_space_len = 0.28

        # # instantiate "length" in net params
        # net_params.additional_params["length"] = \
        #     6 * self.ring_edgelen + 2 * self.intersection_len + \
        #     2 * self.junction_len + 10 * self.inner_space_len

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        r = net_params.additional_params["radius_ring"]

        nodes = [{
            "id": "center",
            "x": 0,
            "y": 0,
            #"radius": 10,
            "radius": (2.9 + 3.3 * net_params.additional_params["lanes"])/2,
            "type": "priority"
        }, {
            "id": "right",
            "x": r,
            "y": 0,
            "type": "priority"
        }, {
            "id": "top",
            "x": 0,
            "y": r,
            "type": "priority"
        }, {
            "id": "left",
            "x": -r,
            "y": 0,
            "type": "priority"
        }, {
            "id": "bottom",
            "x": 0,
            "y": -r,
            "type": "priority"
        }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        r = net_params.additional_params["radius_ring"]
        resolution = net_params.additional_params["resolution"]
        ring_edgelen = 3 * r * pi / 2.
        intersection_edgelen = 2 * r

        # intersection edges
        edges = [{
            "id": "b_c",
            "type": "edgeType",
            #"priority": "78",
            "from": "bottom",
            "to": "center",
            "length": intersection_edgelen / 2
        }, {
            "id": "c_t",
            "type": "edgeType",
            #"priority": 78,
            "from": "center",
            "to": "top",
            "length": intersection_edgelen / 2
        }, {
            "id": "r_c",
            "type": "edgeType",
            #"priority": 78,
            "from": "right",
            "to": "center",
            "length": intersection_edgelen / 2
        }, {
            "id": "c_l",
            "type": "edgeType",
            #"priority": 46,
            "from": "center",
            "to": "left",
            "length": intersection_edgelen / 2
        }, {
            "id": "t_c",
            "type": "edgeType",
            #"priority": 78,
            "from": "top",
            "to": "center",
            "length": intersection_edgelen / 2
        }, {
            "id": "c_r",
            "type": "edgeType",
            #"priority": 46,
            "from": "center",
            "to": "right",
            "length": intersection_edgelen / 2
        }, {
            "id": "l_c",
            "type": "edgeType",
            #"priority": 78,
            "from": "left",
            "to": "center",
            "length": intersection_edgelen / 2
        }, {
            "id": "c_b",
            "type": "edgeType",
            #"priority": "78",
            "from": "center",
            "to": "bottom",
            "length": intersection_edgelen / 2
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
                [(["b_c", "c_t"], 1/3), (["b_c", "c_l"], 1/3),
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
                ["b_c", "c_l"]
        }

        return rts
