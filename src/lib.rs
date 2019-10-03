extern crate regex;
#[macro_use]
extern crate lazy_static;

pub mod einsum;
pub mod ndarray;

pub use einsum::einsum;
pub use ndarray::NDArray;
