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
    ///     tokenizer_name (str, optional): Name of the tokenizer (e.g., "gpt-4").
    ///     tokenizer_path (str, optional): Path to a tokenizer.json file.
    ///     max_tokens (int, optional): Maximum tokens per chunk.
    ///     max_characters (int, optional): Legacy argument. Maximum characters per chunk.
    ///
    /// Note:
    ///     Either (tokenizer_name/path + max_tokens) OR (max_characters) must be provided.
    #[new]
    #[pyo3(signature = (tokenizer_name=None, tokenizer_path=None, max_tokens=None, max_characters=None))]
    pub fn new(
        tokenizer_name: Option<String>, 
        tokenizer_path: Option<String>, 
        max_tokens: Option<usize>,
        max_characters: Option<usize>
    ) -> PyResult<Self> {
        use std::sync::Arc;
        use crate::tokenization::FerrousTokenizer;

        // Legacy path
        if let Some(chars) = max_characters {
            // Warn about deprecation? PyO3 warning utils are tricky, skipping for now.
            return Ok(Self {
                inner: MarkdownChunker::new(chars)
            });
        }

        // Tokenizer path
        let limit = max_tokens.ok_or_else(|| {
             pyo3::exceptions::PyValueError::new_err("Must provide max_tokens or max_characters")
        })?;

        let tokenizer = if let Some(name) = tokenizer_name {
            FerrousTokenizer::new_from_name(&name)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("Unknown tokenizer name: {}", name)))?
        } else if let Some(path) = tokenizer_path {
            FerrousTokenizer::new_from_file(&path)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("Failed to load tokenizer from file: {}", path)))?
        } else {
             return Err(pyo3::exceptions::PyValueError::new_err("Must provide tokenizer_name, tokenizer_path, or max_characters"));
        };

        Ok(Self {
            inner: MarkdownChunker::with_tokenizer(limit, Arc::new(tokenizer))
        })
    }

    /// Chunks a markdown string.
    /// 
    /// Returns:
    ///     list[str]: A list of chunks.
    /// Returns:
    ///     list[str]: A list of chunks.
    pub fn chunk(&self, py: Python<'_>, md: String) -> Vec<String> {
        py.detach(|| self.inner.chunk(&md))
    }

    /// Chunks multiple documents in parallel.
    pub fn chunk_batch(&self, py: Python<'_>, docs: Vec<String>) -> Vec<Vec<String>> {
        py.detach(|| self.inner.chunk_batch(docs))
    }
}
