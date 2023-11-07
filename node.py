from dataclasses import dataclass
from typing import Callable
from uuid import uuid4
from Option import Null, Option, Some

@dataclass
class NodeEdge:
    """
    An "Edge" in graph terms between two nodes

    Raises:
        ValueError: If a negative weight is give
    """
    id: str
    node_a: "Node"
    node_b: "Node"
    weight: float


    def __post_init__(self):
        if not 0 <= self.weight:
            raise ValueError(f"Invalid weight: {self.weight}")

    def __repr__(self) -> str:
        return f"{self.node_a.id}<-{self.weight}->{self.node_b.id}"

@dataclass
class Node:
    id: str
    lat: float
    lon: float


    def __post_init__(self):
        if not -90 <= self.lat <= 90:
            raise ValueError(f"Invalid latitude {self.lat}")
        if not -180 <= self.lon <= 180:
            raise ValueError(f"Invalid longitude {self.lon}")

    def __repr__(self) -> str:
        return f"Node({self.id}@({self.lat},{self.lon}))"

@dataclass
class Species:
    name: str
    species_data: str #placeholder

@dataclass
class Plants:
    species: Species


class Map:
    def __init__(self, id_gen: Callable[[], str]=lambda: str(uuid4())) -> None:
        self.id_generator: Callable[[], str] = id_gen
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, NodeEdge] = {}

    def __repr__(self) -> str:
        return f"""Map:

    Nodes: {self.nodes}

    Edges: {self.edges}"""

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> Option[Node]:
        if node_id in self.nodes.keys():
            return Some(self.nodes[node_id])
        return Null

    def remove_node(self, node_id: str) -> None:
        to_remove = [edge.id for edge in self.edges.values() if edge.node_a.id == node_id or edge.node_b.id == node_id]
        for edge_id in to_remove:
            self.remove_edge(edge_id)
        del self.nodes[node_id]
        # TODO Silent errors

    def add_edge(self, id_a: str, id_b: str, weight: float) -> None:
        if self.get_edge(id_a, id_b).is_some():
            raise Exception("Edge exists")
            # TODO Either update edge, do nothing, 

        match (self.get_node(id_a), self.get_node(id_b)):
            case (Some(a), Some(b)):
                edge_id = self.id_generator()
                self.edges[edge_id] = NodeEdge(edge_id, a, b, weight)
            case _:
                # Error handling
                pass

    def get_edge(self, id_a: str, id_b: str) -> Option[NodeEdge]:
        for edge in self.edges.values():
            a, b = edge.node_a.id, edge.node_b.id
            if (a == id_a and b == id_b) or (a == id_b and b == id_a):
                return Some(edge)
        return Null


    def remove_edge_by_nodes(self, id_a: str, id_b: str) -> None:
        match self.get_edge(id_a, id_b):
            case Some(e):
                self.remove_edge(e.id)
            case _:
                # Error Handling
                pass

    def remove_edge(self, edge_id: str) -> None:
        del self.edges[edge_id]
        # TODO Silent errors
