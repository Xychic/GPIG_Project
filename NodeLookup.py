from typing import Callable
from node import NodeEdge, Lat, Lon
import node_lookup

def get_path(edges: dict[str, NodeEdge], start: str, end: str, heuristic: Callable[[tuple[Lat, Lon], tuple[Lat, Lon]], float]) -> tuple[float, list[str]]:
    return node_lookup.get_path(edges, start, end, heuristic) # type: ignore
