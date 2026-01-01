use pyo3::prelude::*;
use crate::packing::packer::{ContextPacker, PackingStrategy};

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
    ///     strategy (str, optional): Packing strategy - "ranked" (default) or "u_shaped".
    ///         The "u_shaped" strategy places important content at the start and end
    ///         of the context window, based on research showing LLMs attend less to
    ///         content in the middle ("Lost in the Middle", Liu et al., 2023).
    #[new]
    #[pyo3(signature = (max_chars, strategy=None))]
    pub fn new(max_chars: usize, strategy: Option<String>) -> PyResult<Self> {
        let packing_strategy = match strategy.as_deref() {
            None | Some("ranked") => PackingStrategy::Ranked,
            Some("u_shaped") => PackingStrategy::UShaped,
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unknown strategy '{}'. Use 'ranked' or 'u_shaped'.", other)
                ));
            }
        };
        
        Ok(Self {
            inner: ContextPacker::with_strategy(max_chars, packing_strategy),
        })
    }

    /// Packs multiple documents into a single dense context string.
    /// 
    /// Args:
    ///     documents (list[str]): List of retrieved document strings.
    /// 
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
