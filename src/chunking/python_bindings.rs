use pyo3::prelude::*;
use crate::chunking::MarkdownChunker;

/// Python-exposed class for structure-aware Markdown chunking.
#[pyclass]
pub struct PyMarkdownChunker {
    inner: MarkdownChunker,
}

#[pymethods]
impl PyMarkdownChunker {
    /// Create a new MarkdownChunker.
    /// 
    /// Args:
    ///     max_tokens (int): Maximum length of each chunk.
    #[new]
    pub fn new(max_tokens: usize) -> Self {
        Self {
            inner: MarkdownChunker::new(max_tokens),
        }
    }

    /// Chunks a markdown string.
    /// 
    /// Returns:
    ///     list[str]: A list of chunks.
    pub fn chunk(&self, md: &str) -> Vec<String> {
        self.inner.chunk(md)
    }
}
