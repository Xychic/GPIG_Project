from dataclasses import dataclass
from typing import List


@dataclass
class Node:
    id: int
    lat: float
    lon: float
    connected_nodes: List[Node]

    def __post_init__(self):
        if not -90 <= self.lat <= 90:
            raise ValueError(f"Invalid latitude {self.lat}")
        if not -180 <= self.lon <= 180:
            raise ValueError(f"Invalid longitude {self.lon}")
        

@dataclass
class Species:
    name: str
    speciesdata: str #placeholder

@dataclass
class Plants:
    species : Species
    