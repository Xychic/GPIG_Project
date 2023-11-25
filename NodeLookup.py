from typing import Callable
from node import NodeEdge
import node_lookup

def get_path(edges: dict[str, NodeEdge], start: str, end: str, heuristic: Callable[[tuple[float, float], tuple[float, float]], float]) -> tuple[float, list[str]]:
    return node_lookup.get_path(edges, start, end, heuristic) # type: ignore