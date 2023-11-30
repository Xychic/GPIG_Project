use std::{collections::HashSet, hash::Hash};

use cpython::{FromPyObject, ObjectProtocol, PyDict, PyObject, PyResult, Python};

use super::node::Node;

#[derive(Debug, PartialEq)]
pub struct NodeEdge {
    pub id: String,
    pub node_a: Node,
    pub node_b: Node,
    pub weight: f64,
}

impl Eq for NodeEdge {}

impl Hash for NodeEdge {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.node_a.hash(state);
        self.node_b.hash(state);
        ((self.weight * 1E6) as isize).hash(state);
    }
}

impl NodeEdge {
    pub fn from_py_dict(py: Python, edge_dict: PyObject) -> HashSet<NodeEdge> {
        let mut data = HashSet::new();
        let edge_dict: PyDict = edge_dict.extract(py).unwrap();
        for (_, v) in edge_dict.items(py) {
            data.insert(v.extract(py).unwrap());
        }

        data
    }
}

impl<'s> FromPyObject<'s> for NodeEdge {
    fn extract(py: Python, obj: &'s PyObject) -> PyResult<Self> {
        Ok(NodeEdge {
            id: obj.getattr(py, "id").unwrap().extract(py).unwrap(),
            node_a: obj.getattr(py, "node_a").unwrap().extract(py).unwrap(),
            node_b: obj.getattr(py, "node_b").unwrap().extract(py).unwrap(),
            weight: obj.getattr(py, "weight").unwrap().extract(py).unwrap(),
        })
    }
}
