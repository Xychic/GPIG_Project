import IdDictionary
from objects.node import Map, Node, NodeEdge , distanceBetweenNodes
from AStar import AStar
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


tree_map = Map(idGen.gen_id)
tree_map.add_node(a)
tree_map.add_node(b)
tree_map.add_node(c)
tree_map.add_node(d)

tree_map.add_edge("A", "B", 1)
tree_map.add_edge("B", "C", 2)
tree_map.add_edge("C", "D", 3)
tree_map.add_edge("D", "A", 4)

print(tree_map)

print(AStar(a,d,tree_map))
#Distance Calculation
centralHallNode = Node("Central Hall",53.94703,-1.05284)
compSciNode = Node("Dep of Computer Science",53.94682,-1.03086)

print(f"Distance between{centralHallNode} and {compSciNode} is {distanceBetweenNodes(centralHallNode,compSciNode)} ")