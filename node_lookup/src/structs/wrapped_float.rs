use std::ops::{Add, Sub};

use cpython::{FromPyObject, ObjectProtocol};

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord)]
pub struct WrappedFloat {
    pub data: isize,
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
impl WrappedFloat {
    const SCALE: f32 = 1E6;
    pub const MAX: WrappedFloat = WrappedFloat { data: isize::MAX };

    pub fn new(data: f32) -> Self {
        WrappedFloat {
            data: (data * Self::SCALE).trunc() as isize,
        }
    }

    pub fn value(self) -> f32 {
        self.data as f32 / Self::SCALE
    }
}

impl<'s> FromPyObject<'s> for WrappedFloat {
    fn extract(py: cpython::Python, obj: &'s cpython::PyObject) -> cpython::PyResult<Self> {
        Ok(WrappedFloat {
            data: obj.getattr(py, "data")?.extract(py)?,
        })
    }
}

impl Sub for WrappedFloat {
    type Output = WrappedFloat;

    fn sub(self, rhs: Self) -> Self::Output {
        WrappedFloat {
            data: self.data - rhs.data,
        }
    }
}

impl Add for WrappedFloat {
    type Output = WrappedFloat;

    fn add(self, rhs: Self) -> Self::Output {
        WrappedFloat {
            data: self.data + rhs.data,
        }
    }
}

pub type Lat = WrappedFloat;
pub type Lon = WrappedFloat;
