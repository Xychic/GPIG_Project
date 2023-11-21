use std::hash::Hash;

use cpython::{FromPyObject, ObjectProtocol, PyObject, PyResult, Python};

#[derive(Debug, PartialEq, Clone)]
pub struct Node {
    pub id: String,
    pub lat: f64,
    pub lon: f64,
}

impl Node {
    pub fn dist(&self, other: &Node) -> f64 {
        (self.lat - other.lat).abs() + (self.lon - other.lon).abs()
    }
}

impl Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        ((self.lat * 1E6) as isize).hash(state);
        ((self.lon * 1E6) as isize).hash(state);
    }
}

impl<'s> FromPyObject<'s> for Node {
    fn extract(py: Python, obj: &'s PyObject) -> PyResult<Self> {
        Ok(Node {
            id: obj.getattr(py, "id").unwrap().extract(py).unwrap(),
            lat: obj.getattr(py, "lat").unwrap().extract(py).unwrap(),
            lon: obj.getattr(py, "lon").unwrap().extract(py).unwrap(),
        })
    }
}
