use pyo3::prelude::*;

pub mod cache;
pub mod chunking;
pub mod packing;

/// Core module for ferrous Rust implementations.
/// This module is exposed to Python via the `_ferrous` extension.
#[pymodule]
fn _ferrous(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<cache::FuzzyCache>()?;
    m.add_class::<chunking::PyMarkdownChunker>()?;
    m.add_class::<packing::PyContextPacker>()?;
    m.add_function(wrap_pyfunction!(hello_world, m)?)?;
    Ok(())
}

/// A simple greeting function to verify the Python-Rust bridge.
#[pyfunction]
fn hello_world() -> PyResult<String> {
    Ok("Hello from Ferrous! 🦀".to_string())
}
