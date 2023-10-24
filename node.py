from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Node:
    id: int
    lat: float
    lon: float

    def __post_init__(self):
        if not -90 <= self.lat <= 90:
            raise ValueError(f"Invalid latitude {self.lat}")
        if not -180 <= self.lon <= 180:
            raise ValueError(f"Invalid longitude {self.lon}")

@dataclass 
class NodeEdge:
    """
    An "Edge" in graph terms between two nodes

    Raises:
        ValueError: If a negative weight is give
    """
    node_a: Node
    node_b: Node
    weight: float


    def __post_init__(self):
        if not 0 <= self.weight:
            raise ValueError(f"Invalid weight: {self.weight}")

@dataclass
class Graph:
    """
    """

    nodes: dict[int, Node] = field(default_factory=dict)
    weights: defaultdict[int, list[NodeEdge]] = field(default_factory=lambda: defaultdict(list))

    def add_node(self, n: Node):
        if n.id in self.nodes:
            raise ValueError(f"Graph already contains node: {n}")
        self.nodes[n.id] = n

    def remove_node(self, n: Node):
        if n.id not in self.nodes:
            raise ValueError(f"Graph doesn't contain node: {n}")

        connections = self.weights[n.id]

        while len(connections):
            edge = connections[0]
            self.weights[edge.node_a.id].remove(edge)
            self.weights[edge.node_b.id].remove(edge)
        del(self.weights[n.id])
        del(self.nodes[n.id])


    def add_connection(self, a: Node, b: Node, weight: float):
        if a.id not in self.nodes:
            raise ValueError(f"Node not in graph: {a}")
        if b.id not in self.nodes:
            raise ValueError(f"Node not in graph: {b}")
        if a.id == b.id:
            raise ValueError(f"Node cannot connect to itself")
        if self.get_connection(a, b) is not None:
            raise ValueError(f"Connection already exists")

        edge = NodeEdge(a, b, weight)
        self.weights[a.id].append(edge)
        self.weights[b.id].append(edge)

    def get_connection(self, a: Node, b: Node) -> NodeEdge | None:
        if a.id not in self.nodes:
            raise ValueError(f"Node not in graph: {a}")
        if b.id not in self.nodes:
            raise ValueError(f"Node not in graph: {b}")

        a_connections = self.weights[a.id]
        b_connections = self.weights[b.id]
        if len(a_connections) < len(b_connections):
            connections = a_connections
        else:
            connections = b_connections

        for connection in connections:
            if (connection.node_a.id == a.id and connection.node_b.id == b.id) or \
                connection.node_a.id == b.id and connection.node_b.id == a.id:
                return connection
        return None

    def remove_connection(self, a: Node, b: Node):
        connection = self.get_connection(a, b)
        if connection is None:
            raise ValueError(f"No connection between nodes: {a}, {b}")

        self.weights[a.id].remove(connection)
        self.weights[b.id].remove(connection)



@dataclass
class Species:
    name: str
    species_data: str #placeholder

@dataclass
class Plants:
    species: Species
