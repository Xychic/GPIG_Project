#[macro_use]
extern crate cpython;
mod structs;

use cpython::{PyObject, PyResult, Python};
use structs::node::Node;

fn as_str(py: Python, node: PyObject) -> PyResult<String> {
    let n: Node = node.extract(py).unwrap();
    Ok(format!("{:?}", n))
}

py_module_initializer!(node_lookup, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust")?;
    m.add(py, "as_str", py_fn!(py, as_str(node: PyObject)))?;
    Ok(())
});
