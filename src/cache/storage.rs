use rusqlite::{params, Connection, Result};
use rustc_hash::FxHashMap;
use std::path::Path;

/// Storage backend for the Fuzzy Cache using SQLite.
/// 
/// We store (fingerprint, original_string, embedding_json).
/// This allows us to retrieve a cached "hit" if a similar string is found.
pub struct SQLiteStorage {
    conn: Connection,
}

impl SQLiteStorage {
    /// Opens or creates a new SQLite database at the specified path.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        
        // Initialize the table if it doesn't exist.
        // Band columns enable Pigeonhole fuzzy lookup: if Hamming dist <= 3,
        // at least one 16-bit band must match exactly.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS fuzzy_cache (
                id INTEGER PRIMARY KEY,
                fingerprint INTEGER NOT NULL,
                band_a INTEGER NOT NULL DEFAULT 0,
                band_b INTEGER NOT NULL DEFAULT 0,
                band_c INTEGER NOT NULL DEFAULT 0,
                band_d INTEGER NOT NULL DEFAULT 0,
                input_text TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Primary fingerprint index for exact lookups
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fingerprint ON fuzzy_cache (fingerprint)",
            [],
        )?;
        
        // Pigeonhole band indexes for fuzzy lookups
        conn.execute("CREATE INDEX IF NOT EXISTS idx_band_a ON fuzzy_cache (band_a)", [])?;
        conn.execute("CREATE INDEX IF NOT EXISTS idx_band_b ON fuzzy_cache (band_b)", [])?;
        conn.execute("CREATE INDEX IF NOT EXISTS idx_band_c ON fuzzy_cache (band_c)", [])?;
        conn.execute("CREATE INDEX IF NOT EXISTS idx_band_d ON fuzzy_cache (band_d)", [])?;

        Ok(Self { conn })
    }
    
    /// Splits a 64-bit fingerprint into 4 x 16-bit bands for Pigeonhole indexing.
    #[inline]
    fn split_bands(fp: u64) -> (i64, i64, i64, i64) {
        (
            ((fp >> 48) & 0xFFFF) as i64,
            ((fp >> 32) & 0xFFFF) as i64,
            ((fp >> 16) & 0xFFFF) as i64,
            (fp & 0xFFFF) as i64,
        )
    }

    /// Stores a result in the cache with Pigeonhole band indexing.
    pub fn put(&self, fingerprint: u64, input_text: &str, data: &str) -> Result<()> {
        let (a, b, c, d) = Self::split_bands(fingerprint);
        self.conn.execute(
            "INSERT INTO fuzzy_cache (fingerprint, band_a, band_b, band_c, band_d, input_text, data) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![fingerprint as i64, a, b, c, d, input_text, data],
        )?;
        Ok(())
    }

    /// Finds entries with an exact fingerprint match. 
    /// Note: SimHash can have hits with slightly different fingerprints (within Hamming distance).
    /// For V1, we search for +/- small bit flips or exact matches.
    pub fn get_exact(&self, fingerprint: u64) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT data FROM fuzzy_cache WHERE fingerprint = ?1 LIMIT 1"
        )?;
        let mut rows = stmt.query(params![fingerprint as i64])?;

        if let Some(row) = rows.next()? {
            let data: String = row.get(0)?;
            Ok(Some(data))
        } else {
            Ok(None)
        }
    }

    /// Optimized batch retrieval using a single SQL query.
    pub fn get_exact_batch(&self, fingerprints: &[u64]) -> Result<Vec<Option<String>>> {
        if fingerprints.is_empty() {
            return Ok(vec![]);
        }

        // 1. Deduplicate fingerprints to minimize SQL 'IN' clause size
        let mut unique_fps: Vec<u64> = fingerprints.iter().cloned().collect();
        unique_fps.sort_unstable();
        unique_fps.dedup();

        // 2. Build query dynamically: "SELECT fingerprint, data FROM ... WHERE fingerprint IN (?,?,?) GROUP BY fingerprint"
        let placeholders: Vec<&str> = vec!["?"; unique_fps.len()];
        let query = format!(
            "SELECT fingerprint, data FROM fuzzy_cache WHERE fingerprint IN ({}) GROUP BY fingerprint",
            placeholders.join(",")
        );

        let mut stmt = self.conn.prepare(&query)?;
        
        // 3. Map params - cast to i64 to match SQLite storage format
        let fingerprints_i64: Vec<i64> = unique_fps.iter().map(|f| *f as i64).collect();
        let params: Vec<&dyn rusqlite::ToSql> = fingerprints_i64.iter()
            .map(|f| f as &dyn rusqlite::ToSql)
            .collect();

        // 4. Execute and build map (FxHashMap is faster for integer keys)
        let mut found_map: FxHashMap<u64, String> = FxHashMap::default();
        found_map.reserve(unique_fps.len());
        
        let rows = stmt.query_map(&*params, |row| {
             let f: i64 = row.get(0)?;
             let d: String = row.get(1)?;
             Ok((f, d))
        })?;

        for row in rows {
            let (f, d) = row?;
            found_map.insert(f as u64, d);
        }

        // 5. Return results in original order using the map
        let results: Vec<Option<String>> = fingerprints.iter()
            .map(|fp| found_map.get(fp).cloned()) 
            .collect();

        Ok(results)
    }

    /// Finds the closest match within a Hamming distance threshold.
    /// This is an O(N) operation currently. For massive caches (1M+), 
    /// we would want to use a BK-Tree or Multi-index hashing.
    pub fn find_nearby(&self, fingerprint: u64, threshold: u32) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT fingerprint, data FROM fuzzy_cache"
        )?;
        
        let rows = stmt.query_map([], |row| {
            let f: i64 = row.get(0)?;
            let d: String = row.get(1)?;
            Ok((f as u64, d))
        })?;

        for row in rows {
            let (f, d) = row?;
            let dist = (f ^ fingerprint).count_ones();
            if dist <= threshold {
                return Ok(Some(d));
            }
        }

        Ok(None)
    }

    /// Stores multiple results in the cache using a single transaction.
    /// This is significantly faster than multiple single `put` calls.
    pub fn put_batch(&mut self, items: Vec<(u64, String, String)>) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO fuzzy_cache (fingerprint, band_a, band_b, band_c, band_d, input_text, data) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"
            )?;
            for (fp, text, data) in items {
                let (a, b, c, d) = Self::split_bands(fp);
                stmt.execute(params![fp as i64, a, b, c, d, text, data])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Performs fuzzy search for multiple fingerprints using Deferred Loading.
    /// 
    /// Optimization: We scan only (rowid, fingerprint) first (integers are cheap),
    /// then fetch the heavy `data` string only for matches. This avoids reading
    /// 50k strings when we typically only need ~1-5.
    /// 
    /// Complexity: O(N) for scan + O(M) for fetch where M = number of matches.
    pub fn find_nearby_batch(&self, queries: &[u64], threshold: u32) -> Vec<Option<String>> {
        let mut results = vec![None; queries.len()];
        
        if queries.is_empty() {
            return results;
        }

        // Phase 1: FAST SCAN - Only read rowid and fingerprint (integers)
        let mut stmt = match self.conn.prepare("SELECT rowid, fingerprint FROM fuzzy_cache") {
            Ok(s) => s,
            Err(_) => return results,
        };

        // Track which queries found matches: query_index -> (rowid, distance)
        let mut matches: FxHashMap<usize, (i64, u32)> = FxHashMap::default();
        matches.reserve(queries.len());

        let rows = match stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
        }) {
            Ok(r) => r,
            Err(_) => return results,
        };

        // Phase 2: Distance calculation in Rust (CPU-bound on u64s, very fast)
        for row in rows {
            if let Ok((rowid, fp_i64)) = row {
                let fp = fp_i64 as u64;

                for (i, &target_fp) in queries.iter().enumerate() {
                    // Skip if we already found a match for this query
                    if matches.contains_key(&i) {
                        continue;
                    }

                    let dist = (fp ^ target_fp).count_ones();
                    if dist <= threshold {
                        matches.insert(i, (rowid, dist));
                    }
                }
            }

            // Early exit if all queries found matches
            if matches.len() == queries.len() {
                break;
            }
        }

        if matches.is_empty() {
            return results;
        }

        // Phase 3: SLOW FETCH - Get data only for winning rowids
        let rowids: Vec<i64> = matches.values().map(|(rowid, _)| *rowid).collect();
        let placeholders: Vec<&str> = vec!["?"; rowids.len()];
        let query = format!(
            "SELECT rowid, data FROM fuzzy_cache WHERE rowid IN ({})",
            placeholders.join(",")
        );

        let mut stmt_data = match self.conn.prepare(&query) {
            Ok(s) => s,
            Err(_) => return results,
        };

        let params: Vec<&dyn rusqlite::ToSql> = rowids.iter()
            .map(|id| id as &dyn rusqlite::ToSql)
            .collect();

        let data_rows = match stmt_data.query_map(&*params, |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        }) {
            Ok(r) => r,
            Err(_) => return results,
        };

        // Build rowid -> data map
        let mut data_map: FxHashMap<i64, String> = FxHashMap::default();
        for row in data_rows {
            if let Ok((rowid, data)) = row {
                data_map.insert(rowid, data);
            }
        }

        // Populate results
        for (query_idx, (rowid, _)) in matches {
            if let Some(data) = data_map.remove(&rowid) {
                results[query_idx] = Some(data);
            }
        }

        results
    }
    
    /// Performs fuzzy search using Pigeonhole Principle for O(1) candidate lookup.
    /// 
    /// If two 64-bit hashes have Hamming distance <= 3, at least one of their
    /// four 16-bit bands must match exactly. We use band indexes to find candidates
    /// then verify with full Hamming distance check.
    /// 
    /// Complexity: O(K) where K = number of candidates (~10-50 typically) vs O(N) full scan.
    pub fn find_nearby_pigeonhole(&self, fingerprint: u64, threshold: u32) -> Result<Option<String>> {
        let (a, b, c, d) = Self::split_bands(fingerprint);
        
        // Query using OR on band indexes - SQLite will use index OR optimization
        let mut stmt = self.conn.prepare(
            "SELECT fingerprint, data FROM fuzzy_cache 
             WHERE band_a = ?1 OR band_b = ?2 OR band_c = ?3 OR band_d = ?4"
        )?;
        
        let rows = stmt.query_map(params![a, b, c, d], |row| {
            let f: i64 = row.get(0)?;
            let d: String = row.get(1)?;
            Ok((f as u64, d))
        })?;
        
        // Verify candidates with full Hamming distance check
        for row in rows {
            let (fp, data) = row?;
            let dist = (fp ^ fingerprint).count_ones();
            if dist <= threshold {
                return Ok(Some(data));
            }
        }
        
        Ok(None)
    }
    
    /// Performs batch fuzzy search using Pigeonhole Principle with a SINGLE SQL query.
    /// 
    /// Instead of 100 separate queries, we consolidate into one:
    /// `WHERE band_a IN (...) OR band_b IN (...) OR band_c IN (...) OR band_d IN (...)`
    /// 
    /// This fetches a "Candidate Soup" of ~100-200 rows, which we then filter in Rust.
    /// Complexity: O(1) SQL query + O(N*K) in-memory filtering where K = candidates.
    pub fn find_nearby_batch_pigeonhole(&self, targets: &[u64], threshold: u32) -> Vec<Option<String>> {
        if targets.is_empty() {
            return vec![];
        }

        // 1. Prepare Band Collections
        let mut bands_a: Vec<i64> = Vec::with_capacity(targets.len());
        let mut bands_b: Vec<i64> = Vec::with_capacity(targets.len());
        let mut bands_c: Vec<i64> = Vec::with_capacity(targets.len());
        let mut bands_d: Vec<i64> = Vec::with_capacity(targets.len());

        for &fp in targets {
            bands_a.push(((fp >> 48) & 0xFFFF) as i64);
            bands_b.push(((fp >> 32) & 0xFFFF) as i64);
            bands_c.push(((fp >> 16) & 0xFFFF) as i64);
            bands_d.push((fp & 0xFFFF) as i64);
        }

        // 2. Build ONE Metadata Query (Deferred Loading)
        let q = format!(
            "SELECT rowid, fingerprint FROM fuzzy_cache WHERE 
             band_a IN ({}) OR 
             band_b IN ({}) OR 
             band_c IN ({}) OR 
             band_d IN ({})",
            vec!["?"; bands_a.len()].join(","),
            vec!["?"; bands_b.len()].join(","),
            vec!["?"; bands_c.len()].join(","),
            vec!["?"; bands_d.len()].join(",")
        );

        // 3. Bind Parameters
        let mut params: Vec<&dyn rusqlite::ToSql> = Vec::with_capacity(targets.len() * 4);
        for b in &bands_a { params.push(b); }
        for b in &bands_b { params.push(b); }
        for b in &bands_c { params.push(b); }
        for b in &bands_d { params.push(b); }

        // 4. Execute & Filter Metadata
        let mut stmt = match self.conn.prepare(&q) {
            Ok(s) => s,
            Err(_) => return vec![None; targets.len()],
        };
        
        let mut candidates: FxHashMap<i64, u64> = FxHashMap::default();
        let rows = match stmt.query_map(&*params, |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
        }) {
            Ok(r) => r,
            Err(_) => return vec![None; targets.len()],
        };

        for r in rows {
            if let Ok((id, fp_i64)) = r {
                candidates.insert(id, fp_i64 as u64);
            }
        }

        // Identify which targets matched which candidates (rowid -> targets_indices)
        // matches: rowid -> best_dist
        let mut match_map: FxHashMap<i64, u32> = FxHashMap::default();
        let mut winning_rowids = Vec::new();
        let mut target_to_rowid = vec![None; targets.len()];

        for (i, &target_fp) in targets.iter().enumerate() {
            let mut best_dist = u32::MAX;
            let mut best_rowid = None;

            for (&id, &cand_fp) in &candidates {
                let dist = (target_fp ^ cand_fp).count_ones();
                if dist <= threshold && dist < best_dist {
                    best_dist = dist;
                    best_rowid = Some(id);
                    if dist == 0 { break; }
                }
            }
            
            if let Some(id) = best_rowid {
                target_to_rowid[i] = Some(id);
                if !match_map.contains_key(&id) {
                    match_map.insert(id, best_dist);
                    winning_rowids.push(id);
                }
            }
        }

        if winning_rowids.is_empty() {
            return vec![None; targets.len()];
        }

        // 5. Fetch DATA only for winners
        let placeholders: Vec<&str> = vec!["?"; winning_rowids.len()];
        let fetch_q = format!(
            "SELECT rowid, data FROM fuzzy_cache WHERE rowid IN ({})",
            placeholders.join(",")
        );
        
        let mut fetch_stmt = match self.conn.prepare(&fetch_q) {
            Ok(s) => s,
            Err(_) => return vec![None; targets.len()],
        };

        let fetch_params: Vec<&dyn rusqlite::ToSql> = winning_rowids.iter()
            .map(|id| id as &dyn rusqlite::ToSql)
            .collect();

        let data_rows = match fetch_stmt.query_map(&*fetch_params, |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        }) {
            Ok(r) => r,
            Err(_) => return vec![None; targets.len()],
        };

        let mut data_map: FxHashMap<i64, String> = FxHashMap::default();
        for r in data_rows {
            if let Ok((id, data)) = r {
                data_map.insert(id, data);
            }
        }

        // 6. Build final results
        let mut results = vec![None; targets.len()];
        for i in 0..targets.len() {
            if let Some(rowid) = target_to_rowid[i] {
                results[i] = data_map.get(&rowid).cloned();
            }
        }

        results
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_sqlite_persistence() -> Result<()> {
        let tmp_file = NamedTempFile::new().unwrap();
        let storage = SQLiteStorage::new(tmp_file.path())?;
        
        storage.put(12345, "test input", "{\"val\": 1}")?;
        
        let res = storage.get_exact(12345)?;
        assert!(res.is_some());
        assert_eq!(res.unwrap(), "{\"val\": 1}");
        
        // Test fuzzy find (dist=1)
        let res_near = storage.find_nearby(12344, 1)?; // 12344 is 12345 ^ 1
        assert!(res_near.is_some());
        
        Ok(())
    }
}
