import numpy as np

# east
# south
# north
# west

east_routes = ["edge-west-EW", "edge-north-SN", "edge-south-NS"]
south_routes = ["edge-north-SN", "edge-east-WE", "edge-west-EW"]
north_routes = ["edge-south-NS", "edge-west-EW", "edge-east-WE"]
west_routes = ["edge-east-WE", "edge-south-NS", "edge-north-SN"]

routes_list = []

for num1 in east_routes:
    for num2 in south_routes:
        for num3 in north_routes:
            for num4 in west_routes:
               routes_list.append([num1, num2, num3, num4])
