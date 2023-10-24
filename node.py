from dataclasses import dataclass



@dataclass 
class Node_edge:
    """
    An "Edge" in graph terms between two nodes
    

    Raises:
        ValueError: If a negative weight is give
    """
    node_a:Node
    node_b:Node
    weight:float


    def __post_init__(self):
        if not 0 <= self.weight:
            raise ValueError(f"Invalid weight: {self.weight}")
@dataclass
class Node:
    id: int
    lat: float
    lon: float
    connected_nodes: list["Node_edge"] = None

    def __post_init__(self):
        if not -90 <= self.lat <= 90:
            raise ValueError(f"Invalid latitude {self.lat}")
        if not -180 <= self.lon <= 180:
            raise ValueError(f"Invalid longitude {self.lon}")
    def add_connection(self,node_to_connect ,weight):
        
        """
        Add connection with a node_edge between two nodes.
        This function creates one edge that links both nodes.
        For one edge between nodes this function only needs to be called on one of those nodes.
        """
        #check that the connection does not already exist
        if self.connected_to_node(node_to_connect):
            return
        #create edge/connection
        edge = Node_edge(self,node_to_connect,weight)
        #append to list of connections
        self.connected_nodes.append(edge)
        #add the edge in the other node so both nodes are connected
        node_to_connect.__add_node(edge)

    def __add_node(self,edge):
        """
        Private function to add an edge 
        is called in the add_connection function 
        
        """
        if edge in self.connected_nodes:
            raise ValueError(f"Edge already exists {edge}")
        self.connected_nodes.append(edge)


    def connected_to_node(self,node)->bool:
        """Returns true if this node is connected"""
        for edge in self.connected_nodes:
            if edge.connected_node ==node:
                return True
        return False

@dataclass
class Species:
    name: str
    species_data: str #placeholder

@dataclass
class Plants:
    species: Species
