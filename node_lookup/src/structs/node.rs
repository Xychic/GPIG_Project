use std::hash::Hash;

use cpython::{FromPyObject, ObjectProtocol, PyObject, PyResult, Python};

use super::wrapped_float::{Lat, Lon, WrappedFloat};

#[derive(Debug, PartialEq, Clone)]
pub struct Node {
    pub id: String,
    pub lat: Lat,
    pub lon: Lon,
}

impl Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.lat.hash(state);
        self.lon.hash(state);
    }
}

impl<'s> FromPyObject<'s> for Node {
    fn extract(py: Python, obj: &'s PyObject) -> PyResult<Self> {
        Ok(Node {
            id: obj.getattr(py, "id").unwrap().extract(py).unwrap(),
            lat: WrappedFloat::new(obj.getattr(py, "lat").unwrap().extract(py).unwrap()),
            lon: WrappedFloat::new(obj.getattr(py, "lon").unwrap().extract(py).unwrap()),
        })
    }
}
