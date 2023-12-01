from dataclasses import dataclass, field
from typing import Callable, SupportsFloat, SupportsIndex, TypeAlias
from uuid import uuid4
from Option import Null, Option, Some
import math

ConvertibleToFloat: TypeAlias = str | SupportsFloat | SupportsIndex


class Lat(float):
    def __init__(self, val: ConvertibleToFloat):
        super().__init__()

    def __repr__(self) -> str:
        return f"Lat({float(self)})"


class Lon(float):
    def __init__(self, val: ConvertibleToFloat):
        super().__init__()

    def __repr__(self) -> str:
        return f"Lon({float(self)})"


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
    # TODO why is id a string? Also can we rename from id to something else, clashes with built-in python id
    # TODO need to add site_id here but I don't want to do it incase it breaks something
    id: str
    lat: float
    lon: float
    _neighbour_ids: set[str] = field(default_factory=set)

    # def __post_init__(self):
    #     if not -90 <= self.lat <= 90:
    #         raise ValueError(f"Invalid latitude {self.lat}")
    #     if not -180 <= self.lon <= 180:
    #         raise ValueError(f"Invalid longitude {self.lon}")

    def __repr__(self) -> str:
        return f"Node({self.id}@({self.lat},{self.lon}))"

    def getNeighbours(self) -> set[str]:
        return self._neighbour_ids

    def addNeighbours(self, id: str):
        self._neighbour_ids.add(id)

    def removeNeighbour(self, id: str):
        self._neighbour_ids.remove(id)

    def __hash__(self) -> int:
        return hash((self.id, self.lat, self.lon))


def distanceBetweenNodes(nodeA: Node, nodeB: Node) -> float:
    """Returns the distance in M between
    two nodes calculated from their latitudes
    and longitudes using the haversine formula"""
    # Convert to radians
    Alat: float = math.radians(nodeA.lat)
    Alon: float = math.radians(nodeA.lon)
    Blat: float = math.radians(nodeB.lat)
    Blon: float = math.radians(nodeB.lon)
    latitudeDelta: float = Alat - Blat
    longitudeDelta: float = Alon - Blon
    a: float = (
        math.sin(latitudeDelta / 2) ** 2
        + math.cos(Alat) * math.cos(Blat) * math.sin(longitudeDelta / 2) ** 2
    )
    c: float = 2 * math.asin(math.sqrt(a))
    EarthRadius: float = 6371000
    result: float = EarthRadius * c
    return result


@dataclass
class Species:
    name: str
    species_data: str  # placeholder


@dataclass
class Plants:
    species: Species
