#[macro_use]
extern crate cpython;
mod structs;

use std::collections::{HashSet, HashMap, VecDeque};

use cpython::{PyObject, PyResult, Python, PyErr, exc};
use structs::edge::NodeEdge;

use priority_queue::PriorityQueue;

use crate::structs::map::Map;

fn as_str(py: Python, edge_dict: PyObject) -> PyResult<String> {
    let edges: HashSet<NodeEdge> = NodeEdge::from_py_dict(py, edge_dict);
    Ok(format!("{:?}", edges))
}

fn get_path(
    py: Python,
    edge_dict: PyObject,
    start: PyObject,
    end: PyObject,
    heuristic: PyObject,
) -> PyResult<Vec<String>> {
    // let edges: HashSet<NodeEdge> = NodeEdge::from_py_dict(py, edge_dict);
    let map = Map::from_py_dict(py, edge_dict);
    let start: String = start.extract(py).unwrap();
    let end: String = end.extract(py).unwrap();
    let map_size = map.node_count();
    // let heuristic: &dyn Fn(String, String) -> f64 = heuristic.extract(py).unwrap();
    dbg!(heuristic);

    // unseen // Set of nodes to be evaluated
    let mut unexplored_nodes = PriorityQueue::with_capacity(map_size);
    // seen // set of nodes already seen
    let mut explored_nodes = HashSet::with_capacity(map_size);
    let mut preceding_node = HashMap::with_capacity(map_size);
    let mut cost_to_node = HashMap::with_capacity(map_size);

    // add start node to open
    cost_to_node.insert(start.clone(), 0_f64);
    unexplored_nodes.push(start.clone(), usize::MAX);

    // loop
    // current = pop node with lowest f-cost from open
    while let Some((current, _)) = unexplored_nodes.pop() {
        // add current to seen
        explored_nodes.insert(current.clone());
        // if current is target
        if current == end {
            // return path
            let mut path = VecDeque::with_capacity(*cost_to_node.get(&current).unwrap() as usize);
            let mut path_current = &end;
            path.push_front(path_current.clone());
            while path_current != &start {
                let prev: &String = preceding_node.get(path_current).unwrap();
                path.push_front(prev.clone());
                path_current = prev;
            }
            return Ok(path.into());
        }
        //  for each neighbour to current
        let current_cost = *cost_to_node.get(&current).unwrap_or(&f64::MAX);
        let current_node = map.get_node(&current).unwrap();
        for (neighbour, weight) in map.get_connections(&current).unwrap_or_default() {
            // if neighbour is in seen
            if explored_nodes.contains(neighbour.as_str()) {
                // skip
                continue;
            }
            // let cost = cost_to_node.get(&current).unwrap_or(f64::MAX) + weight;
            let cost = current_cost + weight;
            if &cost < cost_to_node.get(neighbour.as_str()).unwrap_or(&f64::MAX) {
                // if new path to neighbour is shorter or neighbour is not in open
                preceding_node.insert(neighbour.clone(), current.clone());
                cost_to_node.insert(neighbour.clone(), cost);
                let estimated_priority = usize::MAX - (cost + map.get_node(&neighbour).unwrap().dist(current_node)) as usize;
                

                if let Some((_, &old_cost)) = unexplored_nodes.get(&neighbour) {
                    if estimated_priority > old_cost {
                        unexplored_nodes.push(neighbour, estimated_priority);
                    }
                } else {
                    unexplored_nodes.push(neighbour, estimated_priority);
                }
            }
        }
        //             set f-cost of neighbour
        //             set parent of neighbour to current
        //             if neighbour is not in open
        //                 add neighbour to open
    }


    Err(PyErr::new::<exc::LookupError, _>(py, format!("No path from {start} to {end}.")))
}

py_module_initializer!(node_lookup, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust")?;
    m.add(py, "as_str", py_fn!(py, as_str(node: PyObject)))?;
    m.add(
        py,
        "get_path",
        py_fn!(py, get_path(edge_dict: PyObject, start: PyObject, end: PyObject, heuristic: PyObject)),
    )?;
    Ok(())
});
