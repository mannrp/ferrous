use pyo3::prelude::*;
use crate::cache::{SimHash, SQLiteStorage};
use pyo3::exceptions::PyRuntimeError;

/// The FuzzyCache is the high-level Python API for lexical caching.
/// 
/// It combines a SimHash generator for computing fingerprints and 
/// an SQLite database for persistent storage.
#[pyclass]
pub struct FuzzyCache {
    hasher: SimHash,
    storage: SQLiteStorage,
    threshold: u32,
}

#[pymethods]
impl FuzzyCache {
    /// Initialize a new FuzzyCache.
    /// 
    /// Args:
    ///     db_path (str): Path to SQLite database file.
    ///     threshold (int): Hamming distance threshold for "near-duplicates" (default=2).
    ///     shingle_size (int): Size of n-grams for SimHash (default=3).
    #[new]
    #[pyo3(signature = (db_path, threshold=2, shingle_size=3))]
    pub fn new(db_path: &str, threshold: u32, shingle_size: usize) -> PyResult<Self> {
        let storage = SQLiteStorage::new(db_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open database: {}", e)))?;
        
        Ok(Self {
            hasher: SimHash::new(shingle_size),
            storage,
            threshold,
        })
    }

    /// Checks if a similar text exists in the cache and returns its associated data.
    /// 
    /// This is the core "hit/miss" logic. If a hit is found, you save an API call.
    pub fn get(&self, text: &str) -> PyResult<Option<String>> {
        let fingerprint = self.hasher.fingerprint(text);
        
        // Try exact match first (O(log N))
        if let Some(data) = self.storage.get_exact(fingerprint).map_err(|e| PyRuntimeError::new_err(e.to_string()))? {
            return Ok(Some(data));
        }

        // Try fuzzy match if exact fails (O(N) for now)
        let result = self.storage.find_nearby(fingerprint, self.threshold)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(result)
    }

    /// Stores a new text-result pair in the cache.
    pub fn put(&self, text: &str, data: &str) -> PyResult<()> {
        let fingerprint = self.hasher.fingerprint(text);
        self.storage.put(fingerprint, text, data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Computes the raw SimHash fingerprint for debugging purposes.
    pub fn fingerprint(&self, text: &str) -> u64 {
        self.hasher.fingerprint(text)
    }
}
