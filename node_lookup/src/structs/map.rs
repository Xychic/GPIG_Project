use std::collections::HashMap;

use cpython::{PyDict, PyObject, Python};

use super::{node::Node, node_edge::NodeEdge, wrapped_float::WrappedFloat};

#[derive(Debug)]
pub struct Map {
    edges: HashMap<String, NodeEdge>,
    nodes: HashMap<String, Node>,
    connections: HashMap<String, Vec<String>>,
}

impl Map {
    pub fn from_py_dict(py: Python, edge_dict: &PyObject) -> Map {
        let mut connections = HashMap::new();
        let mut edges = HashMap::new();
        let mut nodes = HashMap::new();
        let edge_dict: PyDict = edge_dict.extract(py).unwrap();
        for (_, v) in edge_dict.items(py) {
            let edge: NodeEdge = v.extract(py).unwrap();
            connections
                .entry(edge.node_a.id.clone())
                .or_insert_with(Vec::new)
                .push(edge.id.clone());
            connections
                .entry(edge.node_b.id.clone())
                .or_insert_with(Vec::new)
                .push(edge.id.clone());
            nodes.insert(edge.node_a.id.clone(), edge.node_a.clone());
            nodes.insert(edge.node_b.id.clone(), edge.node_b.clone());
            edges.insert(edge.id.clone(), edge);
        }
        Map {
            edges,
            nodes,
            connections,
        }
    }

    pub fn _get_weight(&self, id_a: &str, id_b: &str) -> Option<WrappedFloat> {
        for edge_id in self.connections.get(id_a)? {
            let edge = self.edges.get(edge_id)?;
            if edge.node_a.id == id_b || edge.node_b.id == id_b {
                return Some(edge.weight);
            }
        }
        None
    }

    pub fn get_connections(&self, node_id: &str) -> Option<Vec<(String, WrappedFloat)>> {
        Some(
            self.connections
                .get(node_id)?
                .iter()
                .map(|k| {
                    let edge = self.edges.get(k).unwrap();
                    (
                        if edge.node_a.id == node_id {
                            edge.node_b.id.clone()
                        } else {
                            edge.node_a.id.clone()
                        },
                        edge.weight,
                    )
                })
                .collect(),
        )
    }

    pub fn get_node(&self, node_id: &str) -> Option<&Node> {
        self.nodes.get(node_id)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}
