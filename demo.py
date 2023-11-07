import IdDictionary
from node import Node

# A node can be made up of any type as we simply use its ID as a reference
# A edge is simply a tuple of 3 things (NodeA, NodeB, Weight) it is bi-directional

idGen = IdDictionary.IdGenerator()

nodeDict = IdDictionary.IdDict[str](idGen.gen_id())
edgeDict = IdDictionary.IdDict[tuple[str, str, int]](idGen.gen_id())

# Create a simple graph of 4 nodes in the shape of a square A -> B -> C -> D -> A
print("Adding the nodes and getting there IDs")
print(f"""Adding node A: {nodeDict.add("A")}""")
print(f"""Adding node B: {nodeDict.add("B")}""")
print(f"""Adding node C: {nodeDict.add("C")}""")
print(f"""Adding node D: {nodeDict.add("D")}""")

print("Adding the edges")
print(edgeDict.add((nodeDict.get_id("A"), nodeDict.get_id("B"), 1)))
print(edgeDict.add((nodeDict.get_id("C"), nodeDict.get_id("B"), 2)))
print(edgeDict.add((nodeDict.get_id("D"), nodeDict.get_id("C"), 3)))
print(edgeDict.add((nodeDict.get_id("A"), nodeDict.get_id("D"), 4)))

for edge_id in edgeDict.get_ids():
    edge = edgeDict.get_obj(edge_id)
    (a, b, weight) = edge
    print(f"{edge} which translates to ({nodeDict.get_obj(a)}, {nodeDict.get_obj(b)}, {weight})")