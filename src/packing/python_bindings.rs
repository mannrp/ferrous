use pyo3::prelude::*;
use crate::packing::packer::ContextPacker;

/// Python-exposed class for intelligent context packing.
/// 
/// It ranks retrieved document sentences by importance (TextRank)
/// and packs them into a token budget, avoiding redundancy.
#[pyclass]
pub struct PyContextPacker {
    inner: ContextPacker,
}

#[pymethods]
impl PyContextPacker {
    /// Create a new ContextPacker.
    /// 
    /// Args:
    ///     max_tokens (int): Maximum total length of the packed context.
    #[new]
    pub fn new(max_tokens: usize) -> Self {
        Self {
            inner: ContextPacker::new(max_tokens),
        }
    }

    /// Packs multiple documents into a single dense context string.
    /// 
    /// Args:
    ///     documents (list[str]): List of retrieved document strings.
    /// 
    /// Returns:
    ///     str: The packed and ranked context.
    /// Returns:
    ///     str: The packed and ranked context.
    pub fn pack(&self, py: Python<'_>, documents: Vec<String>) -> String {
        py.allow_threads(|| self.inner.pack(&documents))
    }
}
