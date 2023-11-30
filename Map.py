from typing import Callable
from uuid import uuid4
from Option import Null, Option, Some

from node import Node, NodeEdge, Lat, Lon


class Map:
    def __init__(self, id_gen: Callable[[], str] = lambda: str(uuid4())):
        self.id_generator: Callable[[], str] = id_gen
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, NodeEdge] = {}
        self.width = 0
        self.height = 0

    def __repr__(self) -> str:
        return f"""Map:

    Nodes: {self.nodes}

    Edges: {self.edges}"""

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes.keys()

    def get_node(self, node_id: str) -> Option[Node]:
        if node_id in self.nodes.keys():
            return Some(self.nodes[node_id])
        return Null

    def remove_node(self, node_id: str):
        to_remove = [
            edge.id
            for edge in self.edges.values()
            if edge.node_a.id == node_id or edge.node_b.id == node_id
        ]
        for edge_id in to_remove:
            self.remove_edge(edge_id)
        del self.nodes[node_id]
        # TODO Silent errors

    def add_edge(self, id_a: str, id_b: str, weight: float):
        if (
            f"{id_a}-{id_b}" in self.edges.keys()
            or f"{id_b}-{id_a}" in self.edges.keys()
        ):
            raise Exception("Edge exists")
        # if self.get_edge(id_a, id_b).is_some():
        #     # TODO Either update edge, do nothing,

        match (self.get_node(id_a), self.get_node(id_b)):
            case (Some(a), Some(b)):
                edge_id = self.id_generator()
                self.edges[edge_id] = NodeEdge(edge_id, a, b, weight)
                # Update the neighbours of the nodes
                a.add_neighbour(id_b)
                b.add_neighbour(id_a)
            case _:
                # Error handling
                pass

    def get_edge(self, id_a: str, id_b: str) -> Option[NodeEdge]:
        for edge in self.edges.values():
            a, b = edge.node_a.id, edge.node_b.id
            if (a == id_a and b == id_b) or (a == id_b and b == id_a):
                return Some(edge)
        return Null

    def get_node_neighbours(self, node_id: str) -> Option[list[str]]:
        """Returns a list of Ids of nodes that are connected to node_id"""
        match self.get_node(node_id):
            case Some(node):
                return Some(sorted(node.get_neighbours()))
            case _:
                # error handling
                return Null

    def remove_edge_by_nodes(self, id_a: str, id_b: str):
        match self.get_edge(id_a, id_b):
            case Some(e):
                self.remove_edge(e.id)
            case _:
                # Error Handling
                pass

    def get_id(self, node: Node) -> Option[str]:
        for id, n in self.nodes.items():
            if n == node:
                return Some(id)
        return Null

    def remove_edge(self, edge_id: str):
        edge: NodeEdge = self.edges[edge_id]
        edge.node_a.remove_neighbour(self.get_id(edge.node_b).unwrap())
        edge.node_b.remove_neighbour(self.get_id(edge.node_a).unwrap())
        del self.edges[edge_id]
        # TODO Silent errors


def from_map(map: str, diagonals: bool = False) -> Map:
    result = Map()
    for y, line in enumerate(map.splitlines()):
        result.height = max(result.height, y)
        for x, char in enumerate(line):
            result.width = max(result.width, x)
            if char != ".":
                continue
            result.add_node(Node(f"{x}:{y}", Lat(x), Lon(y)))
            for dx, dy in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if result.has_node(f"{dx}:{dy}"):
                    result.add_edge(f"{x}:{y}", f"{dx}:{dy}", 10)
            if diagonals:
                for dx, dy in [
                    (x - 1, y + 1),
                    (x - 1, y - 1),
                    (x + 1, y + 1),
                    (x + 1, y - 1),
                ]:
                    if result.has_node(f"{dx}:{dy}"):
                        result.add_edge(f"{x}:{y}", f"{dx}:{dy}", 14)
    return result
