from dataclasses import dataclass

@dataclass
class NodeEdge:
    """
    An "Edge" in graph terms between two nodes

    Raises:
        ValueError: If a negative weight is give
    """
    node_a: "Node"
    node_b: "Node"
    weight: float


    def __post_init__(self):
        if not 0 <= self.weight:
            raise ValueError(f"Invalid weight: {self.weight}")

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
class Species:
    name: str
    species_data: str #placeholder

@dataclass
class Plants:
    species: Species
