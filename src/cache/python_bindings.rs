use pyo3::prelude::*;
use crate::cache::{SimHash, SQLiteStorage};
use pyo3::exceptions::PyRuntimeError;

/// The FuzzyCache is the high-level Python API for lexical caching.
/// 
/// It combines a SimHash generator for computing fingerprints and 
/// an SQLite database for persistent storage.
#[pyclass(unsendable)]
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

    /// Checks multiple texts at once. significantly faster due to reduced FFI and single-pass scan.
    pub fn get_batch(&self, texts: Vec<String>) -> PyResult<Vec<Option<String>>> {
        use rayon::prelude::*;
        
        // Sequential for small batches (Rayon overhead not worth it)
        let hasher = self.hasher.clone();
        let fingerprints: Vec<u64> = if texts.len() < 500 {
            texts.iter().map(|t| hasher.fingerprint(t)).collect()
        } else {
            texts.par_iter().map(|t| hasher.fingerprint(t)).collect()
        };

        // 1. Try exact batch match (O(1) query)
        let mut results = self.storage.get_exact_batch(&fingerprints)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // 2. Identify misses for fuzzy search
        let mut misses_indices = Vec::new();
        let mut misses_fps = Vec::new();

        for (i, res) in results.iter().enumerate() {
            if res.is_none() {
                misses_indices.push(i);
                misses_fps.push(fingerprints[i]);
            }
        }

        // 3. Perform batch fuzzy search on misses using Pigeonhole indexes
        // SAFETY: Chunk to 200 items max (200 * 4 bands = 800 params < 999 SQLite limit)
        const PIGEONHOLE_CHUNK_SIZE: usize = 200;
        
        if !misses_fps.is_empty() {
            for chunk_start in (0..misses_fps.len()).step_by(PIGEONHOLE_CHUNK_SIZE) {
                let chunk_end = (chunk_start + PIGEONHOLE_CHUNK_SIZE).min(misses_fps.len());
                let chunk = &misses_fps[chunk_start..chunk_end];
                
                let fuzzy_results = self.storage.find_nearby_batch_pigeonhole(chunk, self.threshold);
                
                for (j, res) in fuzzy_results.into_iter().enumerate() {
                    if res.is_some() {
                        results[misses_indices[chunk_start + j]] = res;
                    }
                }
            }
        }

        Ok(results)
    }

    /// Stores a new text-result pair in the cache.
    pub fn put(&self, text: &str, data: &str) -> PyResult<()> {
        let fingerprint = self.hasher.fingerprint(text);
        self.storage.put(fingerprint, text, data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Stores multiple text-result pairs in a single transaction.
    pub fn put_batch(&mut self, items: Vec<(String, String)>) -> PyResult<()> {
        use rayon::prelude::*;
        
        // Sequential for small batches (Rayon overhead not worth it)
        let hasher = self.hasher.clone();
        let batch_items: Vec<(u64, String, String)> = if items.len() < 500 {
            items.into_iter()
                .map(|(text, data)| {
                    let fp = hasher.fingerprint(&text);
                    (fp, text, data)
                })
                .collect()
        } else {
            items.into_par_iter()
                .map(|(text, data)| {
                    let fp = hasher.fingerprint(&text);
                    (fp, text, data)
                })
                .collect()
        };

        self.storage.put_batch(batch_items)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Computes the raw SimHash fingerprint for debugging purposes.
    pub fn fingerprint(&self, text: &str) -> u64 {
        self.hasher.fingerprint(text)
    }
}
