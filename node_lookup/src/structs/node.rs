use cpython::{FromPyObject, ObjectProtocol, PyObject, PyResult, Python};


#[derive(Debug)]
pub struct Node {
    id: usize,
    lat: isize,
    lon: isize,
    connected_nodes: Vec<Node>,
}

impl<'s> FromPyObject<'s> for Node {
    fn extract(py: Python, obj: &'s PyObject) -> PyResult<Self> {
        Ok(Node {
            id: obj.getattr(py, "id").unwrap().extract(py).unwrap(),
            lat: obj.getattr(py, "lat").unwrap().extract(py).unwrap(),
            lon: obj.getattr(py, "lon").unwrap().extract(py).unwrap(),
            connected_nodes: Vec::new(),
        })
    }
}
