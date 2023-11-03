import IdDictionary

#A node can be made up of anytype as we simply use its ID as a reference
#A edge is simply a tuple of 3 things (NodeA,NodeB,Weight) it is bi-directional

idGen = IdDictionary.IdGenerator()

nodeDict = IdDictionary.IdDict(idGen.gen_id())
edgeDict = IdDictionary.IdDict(idGen.gen_id())

#create a simple graph of 4 nodes in the shape of a square A -> B -> C -> D -> A
print("Adding the nodes and getting there IDs")
print(f"Adding node A :{nodeDict.add('A')}")
print(f"Adding node B :{nodeDict.add('B')}")
print(f"Adding node C :{nodeDict.add('C')}")
print(f"Adding node D :{nodeDict.add('D')}")

print("Adding the edges")
print(edgeDict.add((nodeDict.get_id("A"),nodeDict.get_id("B"),1)))
print(edgeDict.add((nodeDict.get_id("C"),nodeDict.get_id("B"),1)))
print(edgeDict.add((nodeDict.get_id("D"),nodeDict.get_id("C"),1)))
print(edgeDict.add((nodeDict.get_id("A"),nodeDict.get_id("D"),1)))

for edgeid in  edgeDict.get_ids():
    edge = edgeDict.get_obj(edgeid)
    print(f"{edge} which translates to ({nodeDict.get_obj(edge[0])}, {nodeDict.get_obj(edge[1])}, 1)")