import unittest
from Map import Map
from Option import Null, Some
from node import Lat, Lon, Node, NodeEdge


class TestNodeClass(unittest.TestCase):
    def test_constructor(self):
        node_id = "node_id"
        lat = Lat(1.1)
        lon = Lon(2.2)
        node = Node(node_id, lat, lon)
        self.assertEqual(node.id, node_id)
        self.assertEqual(node.lat, lat)
        self.assertEqual(node.lon, lon)
        self.assertEqual(len(node.get_neighbours()), 0)

    def test_add_neighbour(self):
        node_1_id = "node_1_id"
        lat = Lat(1.1)
        lon = Lon(2.2)
        node = Node(node_1_id, lat, lon)
        node_2_id = "node_2_id"
        node.add_neighbour(node_2_id)
        self.assertTrue(node_2_id in node.get_neighbours())

    def test_remove_neighbour(self):
        node_1_id = "node_1_id"
        lat = Lat(1.1)
        lon = Lon(2.2)
        node = Node(node_1_id, lat, lon)
        node_2_id = "node_2_id"
        node.add_neighbour(node_2_id)
        node.remove_neighbour(node_2_id)
        self.assertTrue(node_2_id not in node.get_neighbours())


class TestNodeEdgeClass(unittest.TestCase):
    def test_constructor(self):
        node_1_id = "node_1_id"
        node_1 = Node(node_1_id, Lat(0), Lon(0))
        node_2_id = "node_2_id"
        node_2 = Node(node_2_id, Lat(1), Lon(1))

        node_edge_id = "node_edge_id"
        node_edge_weight = 3
        node_edge = NodeEdge(node_edge_id, node_1, node_2, node_edge_weight)

        self.assertEqual(node_edge.id, node_edge_id)
        self.assertEqual(node_edge.node_a, node_1)
        self.assertEqual(node_edge.node_b, node_2)
        self.assertEqual(node_edge.weight, node_edge_weight)

    def test_weight_positive_constraint(self):
        node_1_id = "node_1_id"
        node_1 = Node(node_1_id, Lat(0), Lon(0))
        node_2_id = "node_2_id"
        node_2 = Node(node_2_id, Lat(1), Lon(1))

        node_edge_id = "node_edge_id"
        node_edge_weight = -1
        with self.assertRaises(ValueError):
            NodeEdge(node_edge_id, node_1, node_2, node_edge_weight)


class TestMapClass(unittest.TestCase):
    def test_default_empty(self):
        map = Map()

        self.assertEqual(len(map.nodes), 0)
        self.assertEqual(len(map.edges), 0)
        self.assertEqual(map.width, 0)
        self.assertEqual(map.height, 0)

    def test_add_node(self):
        map = Map()
        node_id = "node_id"
        node = Node(node_id, Lat(0), Lon(0))
        map.add_node(node)

        self.assertEqual(map.nodes.get(node_id), node)

    def test_has_node(self):
        map = Map()
        node_id = "node_id"
        node = Node(node_id, Lat(0), Lon(0))

        self.assertFalse(map.has_node(node_id))
        map.add_node(node)
        self.assertTrue(map.has_node(node_id))

    def test_get_node_exists(self):
        map = Map()
        node_id = "node_id"
        node = Node(node_id, Lat(0), Lon(0))
        map.add_node(node)
        self.assertEqual(map.get_node(node_id), Some(node))

    def test_get_node_not_exists(self):
        map = Map()
        node_id = "not_a_node"
        self.assertEqual(map.get_node(node_id), Null)

    def test_remove_node(self):
        map = Map()
        node_id = "node_id"
        node = Node(node_id, Lat(0), Lon(0))

        map.add_node(node)
        self.assertTrue(map.has_node(node_id))
        map.remove_node(node_id)
        self.assertFalse(map.has_node(node_id))

    def test_get_id(self):
        map = Map()
        node_id = "node_id"
        node = Node(node_id, Lat(0), Lon(0))
        map.add_node(node)

        self.assertEqual(map.get_id(node), Some(node_id))

    def test_get_id_not_exists(self):
        map = Map()
        node_id = "node_id"
        node = Node(node_id, Lat(0), Lon(0))

        self.assertEqual(map.get_id(node), Null)

    def test_add_edge(self):
        map = Map()
        node_1_id = "node_1_id"
        node_1 = Node(node_1_id, Lat(0), Lon(0))
        node_2_id = "node_2_id"
        node_2 = Node(node_2_id, Lat(1), Lon(1))

        map.add_node(node_1)
        map.add_node(node_2)

        map.add_edge(node_1_id, node_2_id, 2)
        map_edges = list(map.edges.values())
        self.assertEqual(len(map_edges), 1)
        edge = map_edges[0]
        self.assertEqual(edge.node_a, node_1)
        self.assertEqual(edge.node_b, node_2)

    def test_get_edge(self):
        map = Map()
        node_1_id = "node_1_id"
        node_1 = Node(node_1_id, Lat(0), Lon(0))
        node_2_id = "node_2_id"
        node_2 = Node(node_2_id, Lat(1), Lon(1))

        map.add_node(node_1)
        map.add_node(node_2)

        map.add_edge(node_1_id, node_2_id, 2)
        map_edges = list(map.edges.values())
        self.assertEqual(len(map_edges), 1)
        edge_direct = map_edges[0]
        edge_method = map.get_edge(node_1_id, node_2_id)
        self.assertEqual(Some(edge_direct), edge_method)
        edge_method_swapped = map.get_edge(node_2_id, node_1_id)
        self.assertEqual(Some(edge_direct), edge_method_swapped)

    def test_get_edge_not_exists(self):
        map = Map()

        node_1_id = "node_1_not_id"
        node_2_id = "node_2_not_id"
        self.assertEqual(map.get_edge(node_1_id, node_2_id), Null)

    def test_remove_edge(self):
        map = Map()
        node_1_id = "node_1_id"
        node_1 = Node(node_1_id, Lat(0), Lon(0))
        node_2_id = "node_2_id"
        node_2 = Node(node_2_id, Lat(1), Lon(1))

        map.add_node(node_1)
        map.add_node(node_2)

        map.add_edge(node_1_id, node_2_id, 2)
        map.remove_edge_by_nodes(node_1_id, node_2_id)


if __name__ == "__main__":
    unittest.main()
