#[macro_use]
extern crate cpython;

use cpython::{Python, PyResult, PyObject, FromPyObject, ObjectProtocol};

#[derive(Debug)]
struct Node {
    id: usize,
    lat: isize,
    lon: isize,
    connected_nodes: Vec<Node>,
}

impl<'s> FromPyObject<'s> for Node {
    fn extract(py: Python, obj: &'s PyObject) -> PyResult<Self> {
        Ok(
            Node {
                id: obj.getattr(py, "id").unwrap().extract(py).unwrap(),
                lat: obj.getattr(py, "lat").unwrap().extract(py).unwrap(),
                lon: obj.getattr(py, "lon").unwrap().extract(py).unwrap(),
                connected_nodes: Vec::new(),
            }
        )
    }
}

fn as_str(py: Python, node: PyObject) -> PyResult<String> {
    let n: Node = node.extract(py).unwrap();
    Ok(format!("{:?}", n))
}

fn _count_doubles(_py: Python, val: &str) -> PyResult<u64> {
    let _ = _py;
    let mut total = 0u64;

    let mut chars = val.chars();
    if let Some(mut c1) = chars.next() {
        for c2 in chars {
            if c1 == c2 {
                total += 1;
            }
            c1 = c2;
        }
    }

    Ok(total)
}

py_module_initializer!(node_lookup, |py, m | {
    m.add(py, "__doc__", "This module is implemented in Rust")?;
    m.add(py, "as_str", py_fn!(py, as_str(node: PyObject)))?;
    Ok(())
});