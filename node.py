from dataclasses import dataclass
from typing import Callable
from uuid import uuid4
from Option import Null, Option, Some
import math
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
    _neighbour_ids: set[str] = set()

    def __post_init__(self):
        if not -90 <= self.lat <= 90:
            raise ValueError(f"Invalid latitude {self.lat}")
        if not -180 <= self.lon <= 180:
            raise ValueError(f"Invalid longitude {self.lon}")

    def __repr__(self) -> str:
        return f"Node({self.id}@({self.lat},{self.lon}))"
    def getNeighbours(self)->set[str]:
        return self._neighbour_ids
    def addNeighbours(self,id:str):
        self._neighbour_ids.add(id)
    def removeNeighbour(self,id:str):
        self._neighbour_ids.remove(id)
def distanceBetweenNodes(nodeA:Node,nodeB:Node)-> float:
    """Returns the distance in M between 
    two nodes calculated from their latitudes
    and longitudes using the haversine formula"""
    #Convert to radians
    Alat:float = math.radians(nodeA.lat)
    Alon:float = math.radians(nodeA.lon)
    Blat:float = math.radians(nodeB.lat)
    Blon:float = math.radians(nodeB.lon)
    latitudeDelta:float = Alat - Blat
    longitudeDelta:float = Alon - Blon
    a:float = math.sin(latitudeDelta/2)**2 + math.cos(Alat)*math.cos(Blat) * math.sin(longitudeDelta/2)**2
    c:float = 2* math.asin(math.sqrt(a))
    EarthRadius:float = 6371000
    result:float = EarthRadius * c
    return result



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
                #Update the neighbours of the nodes
                a.addNeighbours(id_b)
                b.addNeighbours(id_a)
            case _:
                # Error handling
                pass

    def get_edge(self, id_a: str, id_b: str) -> Option[NodeEdge]:
        for edge in self.edges.values():
            a, b = edge.node_a.id, edge.node_b.id
            if (a == id_a and b == id_b) or (a == id_b and b == id_a):
                return Some(edge)
        return Null

    def get_node_neighbours(self,node_id:str)-> Option[list[str]]:
        """Returns a list of Ids of nodes that are connected to node_id
        """
        if node_id in self.nodes.keys():
            match self.get_node(node_id):
                case Some(node):
                    
                    return Some(sorted(node.getNeighbours()))
                case _:
                    #error handling

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

    def remove_edge(self, edge_id: str) -> None:
        del self.edges[edge_id]
        # TODO Silent errors
