from typing import Callable
from uuid import uuid4
from Option import Null, Option, Some

from node import Node, NodeEdge, Lat, Lon


class Map:
    def __init__(self, id_gen: Callable[[], str] = lambda: str(uuid4())) -> None:
        self.id_generator: Callable[[], str] = id_gen
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, NodeEdge] = {}
        self.node_keys: set[str] = set()
        self.edge_keys: set[str] = set()
        self.width = 0
        self.height = 0

    def __repr__(self) -> str:
        return f"""Map:

    Nodes: {self.nodes}

    Edges: {self.edges}"""

    def add_node(self, node: Node) -> None:
        self.node_keys.add(node.id)
        self.nodes[node.id] = node

    def has_node(self, node_id: str) -> bool:
        return node_id in self.node_keys

    def get_node(self, node_id: str) -> Option[Node]:
        if node_id in self.nodes.keys():
            return Some(self.nodes[node_id])
        return Null

    def remove_node(self, node_id: str) -> None:
        to_remove = [
            edge.id
            for edge in self.edges.values()
            if edge.node_a.id == node_id or edge.node_b.id == node_id
        ]
        for edge_id in to_remove:
            self.remove_edge(edge_id)
        del self.nodes[node_id]
        # TODO Silent errors

    def add_edge(self, id_a: str, id_b: str, weight: float) -> None:
        if f"{id_a}-{id_b}" in self.edge_keys or f"{id_b}-{id_a}" in self.edge_keys:
            raise Exception("Edge exists")
        # if self.get_edge(id_a, id_b).is_some():
        #     # TODO Either update edge, do nothing,

        match (self.get_node(id_a), self.get_node(id_b)):
            case (Some(a), Some(b)):
                edge_id = self.id_generator()
                self.edges[edge_id] = NodeEdge(edge_id, a, b, weight)
                # Update the neighbours of the nodes
                a.addNeighbours(id_b)
                b.addNeighbours(id_a)
                self.edge_keys.add(f"{id_a}-{id_b}")
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
        if node_id in self.nodes.keys():
            match self.get_node(node_id):
                case Some(node):
                    return Some(sorted(node.getNeighbours()))
                case _:
                    # error handling

                    return Null
        else:
            return Null

    def remove_edge_by_nodes(self, id_a: str, id_b: str) -> None:
        match self.get_edge(id_a, id_b):
            case Some(e):
                self.remove_edge(e.id)
                match (self.get_node(id_a), self.get_node(id_b)):
                    case (Some(a), Some(b)):
                        a.removeNeighbour(id_b)
                        b.removeNeighbour(id_a)
                    case _:
                        # Error Handling
                        pass
            case _:
                # Error Handling
                pass

    def getID(self, node: Node) -> str:
        for k in self.nodes:
            if self.nodes[k] == node:
                return k
        raise LookupError

    def remove_edge(self, edge_id: str) -> None:
        edge: NodeEdge = self.edges[edge_id]
        edge.node_a.removeNeighbour(self.getID(edge.node_b))
        edge.node_b.removeNeighbour(self.getID(edge.node_a))
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
                for dx, dy in [(x-1, y-1), (x+1, y+1), (x-1, y+1), (x+1, y-1)]:
                    if result.has_node(f"{dx}:{dy}"):
                        result.add_edge(f"{x}:{y}", f"{dx}:{dy}", 14)
    return result
