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
    ///     max_chars (int): Maximum total length (in chars) of the packed context.
    #[new]
    pub fn new(max_chars: usize) -> Self {
        Self {
            inner: ContextPacker::new(max_chars),
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
        py.detach(|| self.inner.pack(&documents))
    }

    /// Packs multiple batches of documents in parallel.
    pub fn pack_batch(&self, py: Python<'_>, document_sets: Vec<Vec<String>>) -> Vec<String> {
        py.detach(|| self.inner.pack_batch(document_sets))
    }
}
