import unittest


import sys
import os

# Add the path to the directory containing 'node.py' to the system path
node_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', 'objects'))
sys.path.append(node_dir)

# Now you can import 'node'
import node
from node import Node , distanceBetweenNodes
class Test_DistanceBetweenNodes(unittest.TestCase):

    def test_1m_distance(self):
        a = Node("York",53.95864,-1.1217906)
        b = Node("New York",40.3744772,-74.202124)
        self.assertAlmostEqual(5463885,node.distanceBetweenNodes(a,b),places=0,msg="Check accurate to 1m")
    
    def test_distance_between_same_Point(self):
        # Test for distance between same points (should be 0)
        nodeA = Node("A",37.7749, -122.4194)
        nodeB = Node("B",37.7749, -122.4194)
        result = distanceBetweenNodes(nodeA, nodeB)
        self.assertAlmostEqual(result, 0.0, places=2)

class Test_Node(unittest.TestCase):
    def setUp(self):
        self.testNode = Node("York",53.95864,-1.1217906)
    def testGetNeighbours(self):
        self.assertTrue(len(self.testNode.getNeighbours()) ==0,msg="Should be no neighbours")
    
    def test_neighbours(self):
        b = Node("New York",40.3744772,-74.202124)
        c = Node("Newer York",45.3744772,-74.202124)
        self.testNode.addNeighbours(b.id)
        self.testNode.addNeighbours(c.id)
        self.assertIn(b.id,self.testNode.getNeighbours())
        self.assertIn(c.id,self.testNode.getNeighbours())
        self.testNode.removeNeighbour(c.id)
        self.testNode.removeNeighbour(b.id)
        self.assertNotIn(b.id,self.testNode.getNeighbours())
        self.assertNotIn(c.id,self.testNode.getNeighbours())