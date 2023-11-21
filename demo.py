import math
from multiprocessing.util import is_abstract_socket_namespace
import IdDictionary
from Option import Some
from node import Node, NodeEdge , distanceBetweenNodes
from AStar import AStar
import Map
import PrintColours

import node_lookup

# A node can be made up of any type as we simply use its ID as a reference
# A edge is simply a tuple of 3 things (NodeA, NodeB, Weight) it is bi-directional

idGen = IdDictionary.IdGenerator()

nodeDict = IdDictionary.IdDict[Node](idGen.gen_id())
edgeDict = IdDictionary.IdDict[NodeEdge](idGen.gen_id())

# Create a simple graph of 4 nodes in the shape of a square A <-> B <-> C <-> D <-> A
a = Node("A", .0, .0)
b = Node("B", .0, .0)
c = Node("C", .0, .0)
d = Node("D", .0, .0)

print("Adding the nodes and getting there IDs")
print(f"""Adding node A: {nodeDict.add(a)}""")
print(f"""Adding node B: {nodeDict.add(b)}""")
print(f"""Adding node C: {nodeDict.add(c)}""")
print(f"""Adding node D: {nodeDict.add(d)}""")

print("Adding the edges")
print(edgeDict.add(NodeEdge("1", a, b, 1)))
print(edgeDict.add(NodeEdge("2", b, c, 2)))
print(edgeDict.add(NodeEdge("3", c, d, 3)))
print(edgeDict.add(NodeEdge("4", d, a, 4)))

for edge_id in edgeDict.get_ids():
    edge = edgeDict.get_obj(edge_id)
    print(f"{edge} which translates to ({edge.node_a}, {edge.node_b}, {edge.weight})")



data = open("maze-32-32-4.map").read()

tree_map = Map.from_map("\n".join(data.splitlines()[4:]))


def show_path(path: list[str]):
    for y in range(32):
        for x in range(32):
            node_id = f"{x}:{y}"
            if tree_map.get_node(node_id).is_some():
                if node_id in path:
                    print(f"{PrintColours.FAIL}x{PrintColours.ENDC}", end="")
                else:
                    print(".", end="")
            else:
                print(f"{PrintColours.OKBLUE}@{PrintColours.ENDC}", end="")
        print()
    print(f"Total length: {len(path)}")

start = "9:6"
end = "11:21"
print(start, end)

path = set(AStar(tree_map.get_node(start).unwrap(), tree_map.get_node(end).unwrap(), tree_map, lambda a, b: abs(a.lat - b.lat) + abs(a.lon - b.lon)))

show_path([p.id for p in path])
print(path)

def heuristic(a: tuple[float, float], b: tuple[float, float]) -> int:
    (a_lat, a_lon) = a
    (b_lat, b_lon) = b
    return int(abs(a_lat - b_lat) + abs(a_lon - b_lon))

def heuristic_b(a: tuple[float, float], b: tuple[float, float]) -> int:
    (a_lat, a_lon) = a
    (b_lat, b_lon) = b
    Alat:float = math.radians(a_lat)
    Alon:float = math.radians(a_lon)
    Blat:float = math.radians(b_lat)
    Blon:float = math.radians(b_lon)
    latitudeDelta:float = Alat - Blat
    longitudeDelta:float = Alon - Blon
    x:float = math.sin(latitudeDelta/2)**2 + math.cos(Alat)*math.cos(Blat) * math.sin(longitudeDelta/2)**2
    c:float = 2* math.asin(math.sqrt(x))
    EarthRadius:float = 6371000
    result:float = EarthRadius * c
    return int(result)


rust_path = node_lookup.get_path(tree_map.edges, start, end, heuristic)
show_path(rust_path)
#Distance Calculation
centralHallNode = Node("Central Hall",53.94703,-1.05284)
compSciNode = Node("Dep of Computer Science",53.94682,-1.03086)


print(f"Distance between{centralHallNode} and {compSciNode} is {distanceBetweenNodes(centralHallNode,compSciNode)} ")