from dataclasses import dataclass,field
from objects.node import Map,Node,distanceBetweenNodes
from typing import Callable
from queue import PriorityQueue



@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Node=field(compare=False)



    


def AStar(StartNode:Node,GoalNode:Node,graph:Map,Heuristic:Callable[[Node,Node],float]=distanceBetweenNodes)->list[Node]:

    def reconstruct_path(current:Node)->list[Node]:
        route: list[Node]=[current]
        while current in preceding_node.keys():
            current= preceding_node[current]
            route.insert(0,current)
        return route


    

    id = graph.getID
    
    explored_nodes:set[Node] = set()
    unexplored_nodes:PriorityQueue[PrioritizedItem] = PriorityQueue()
    preceding_node:dict[Node,Node] = {}
    cost_to_node:dict[Node,float]={}

    #for all nodes set their cost_to_node as infinite
    for x in graph.nodes.values():
        cost_to_node[x] = float('inf')
    cost_to_node[StartNode] = 0
    

    unexplored_nodes.put(PrioritizedItem(0,StartNode))
    

    while (not unexplored_nodes.empty()):
        current_node:Node = unexplored_nodes.get().item
        if current_node == GoalNode:
            
            return reconstruct_path(current_node)
        explored_nodes.add(current_node)
        for node in current_node.getNeighbours():
            neighbour:Node = graph.get_node(node).unwrap()
            cost:float = cost_to_node[current_node] + graph.get_edge(id(current_node),id(neighbour)).unwrap().weight

            if cost < cost_to_node[neighbour]:
                #if path is better
                preceding_node[neighbour]= current_node
                cost_to_node[neighbour] = cost
                estimated_cost_to_goal_via_node:float = cost_to_node[neighbour] + Heuristic(neighbour,GoalNode)

                #Update the priority queue
                
                #If neighbour already exists update the priority if we have a shorter path to neighbour
                if any(entry.item == neighbour for entry in unexplored_nodes.queue):
                    entry:PrioritizedItem = [obj for obj in unexplored_nodes.queue if obj.item == neighbour][0]
                    if entry.priority <estimated_cost_to_goal_via_node:
                        entry.priority = estimated_cost_to_goal_via_node
                #If neighbour not in the priority queue add it
                else:
                    unexplored_nodes.put(PrioritizedItem(estimated_cost_to_goal_via_node,neighbour))
                
                


    raise LookupError("No valid path")


