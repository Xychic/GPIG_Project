use std::hash::Hash;

use cpython::{FromPyObject, ObjectProtocol, PyObject, PyResult, Python};

#[derive(Debug, PartialEq, Clone)]
pub struct Node {
    pub id: String,
    pub lat: f32,
    pub lon: f32,
}

impl Node {
    pub fn _dist(&self, other: &Node) -> f32 {
        (self.lat - other.lat).abs() + (self.lon - other.lon).abs()
    }
}

impl Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.lat.to_bits().hash(state);
        self.lon.to_bits().hash(state);
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
