#[macro_use]
extern crate cpython;
mod structs;

use std::collections::{HashMap, HashSet, VecDeque};

use cpython::{exc, ObjectProtocol, PyErr, PyObject, PyResult, Python};
use structs::node_edge::NodeEdge;

use priority_queue::PriorityQueue;

use crate::structs::map::Map;

#[allow(clippy::unnecessary_wraps)]
fn as_str(py: Python, edge_dict: &PyObject) -> PyResult<String> {
    let edges: HashSet<NodeEdge> = NodeEdge::from_py_dict(py, edge_dict);
    Ok(format!("{edges:?}"))
}

#[allow(
    clippy::unnecessary_wraps,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn get_path(
    py: Python,
    edge_dict: &PyObject,
    start: &PyObject,
    end: &PyObject,
    heuristic_fn: &PyObject,
) -> PyResult<(f32, Vec<String>)> {
    // let edges: HashSet<NodeEdge> = NodeEdge::from_py_dict(py, edge_dict);
    let map = Map::from_py_dict(py, edge_dict);
    let start: String = start.extract(py).unwrap();
    let end: String = end.extract(py).unwrap();
    let map_size = map.node_count();
    let heuristic: &dyn Fn((f32, f32), (f32, f32)) -> f32 = &|x, y| {
        heuristic_fn
            .call(py, (x, y), None)
            .unwrap()
            .extract(py)
            .unwrap()
    };

    let mut unexplored_nodes = PriorityQueue::with_capacity(map_size);
    let mut explored_nodes = HashSet::with_capacity(map_size);
    let mut preceding_node: HashMap<_, String> = HashMap::with_capacity(map_size);
    let mut cost_to_node = HashMap::with_capacity(map_size);

    cost_to_node.insert(start.clone(), 0_f32);
    unexplored_nodes.push(start.clone(), usize::MAX);

    while let Some((current, _)) = unexplored_nodes.pop() {
        explored_nodes.insert(current.clone());
        if current == end {
            let mut path =
                VecDeque::with_capacity(cost_to_node.get(&current).unwrap().ceil() as usize);
            let mut path_current = &end;
            path.push_front(path_current.clone());
            while path_current != &start {
                let prev = preceding_node.get(path_current).unwrap();
                path.push_front(prev.clone());
                path_current = prev;
            }
            return Ok((*cost_to_node.get(&end).unwrap(), path.into()));
        }
        let current_cost = *cost_to_node.get(&current).unwrap_or(&f32::MAX);
        let current_node = {
            match map.get_node(&current) {
                Some(n) => n,
                None => {
                    return Err(PyErr::new::<exc::LookupError, _>(
                        py,
                        format!("No path from {start} to {end}."),
                    ))
                }
            }
        };

        for (neighbour, weight) in map.get_connections(&current).unwrap_or_default() {
            if explored_nodes.contains(neighbour.as_str()) {
                continue;
            }
            let cost = current_cost + weight;
            if &cost < cost_to_node.get(neighbour.as_str()).unwrap_or(&f32::MAX) {
                preceding_node.insert(neighbour.clone(), current.clone());
                cost_to_node.insert(neighbour.clone(), cost);
                let neighbour_node = map.get_node(&neighbour).unwrap();
                let estimated_priority = usize::MAX
                    - (cost
                        + heuristic(
                            (current_node.lat, current_node.lon),
                            (neighbour_node.lat, neighbour_node.lon),
                        ))
                    .ceil() as usize;

                if let Some((_, &old_cost)) = unexplored_nodes.get(&neighbour) {
                    if estimated_priority > old_cost {
                        unexplored_nodes.push(neighbour, estimated_priority);
                    }
                } else {
                    unexplored_nodes.push(neighbour, estimated_priority);
                }
            }
        }
    }

    Err(PyErr::new::<exc::LookupError, _>(
        py,
        format!("No path from {start} to {end}."),
    ))
}

py_module_initializer!(node_lookup, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust")?;
    m.add(py, "as_str", py_fn!(py, as_str(node: &PyObject)))?;
    m.add(
        py,
        "get_path",
        py_fn!(py, get_path(edge_dict: &PyObject, start: &PyObject, end: &PyObject, heuristic: &PyObject)),
    )?;
    Ok(())
});
