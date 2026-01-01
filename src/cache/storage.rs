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
        // We index the fingerprint for fast O(log N) lookups.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS fuzzy_cache (
                id INTEGER PRIMARY KEY,
                fingerprint INTEGER NOT NULL,
                input_text TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fingerprint ON fuzzy_cache (fingerprint)",
            [],
        )?;

        Ok(Self { conn })
    }

    /// Stores a result in the cache.
    pub fn put(&self, fingerprint: u64, input_text: &str, data: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO fuzzy_cache (fingerprint, input_text, data) VALUES (?1, ?2, ?3)",
            params![fingerprint as i64, input_text, data],
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

        // 1. Build query dynamically: "SELECT fingerprint, data FROM ... WHERE fingerprint IN (?,?,?)"
        // Note: SQLite limits the number of variables (usually 999 or 32766). 
        // We should chunk this if it exceeds limits, but for v0.2 we assume sensible batch sizes (<900).
        let placeholders: Vec<&str> = vec!["?"; fingerprints.len()];
        let query = format!(
            "SELECT fingerprint, data FROM fuzzy_cache WHERE fingerprint IN ({})",
            placeholders.join(",")
        );

        let mut stmt = self.conn.prepare(&query)?;
        
        // 2. Map params - cast to i64 to match SQLite storage format
        let fingerprints_i64: Vec<i64> = fingerprints.iter().map(|f| *f as i64).collect();
        let params: Vec<&dyn rusqlite::ToSql> = fingerprints_i64.iter()
            .map(|f| f as &dyn rusqlite::ToSql)
            .collect();

        // 3. Execute and build map (FxHashMap is faster for integer keys)
        let mut found_map: FxHashMap<u64, String> = FxHashMap::default();
        found_map.reserve(fingerprints.len());
        
        // Use query_map to iterate rows
        let rows = stmt.query_map(&*params, |row| {
             let f: i64 = row.get(0)?;
             let d: String = row.get(1)?;
             Ok((f, d))
        })?;

        for row in rows {
            let (f, d) = row?;
            found_map.insert(f as u64, d);
        }

        // 4. Return results in order
        let results = fingerprints.iter()
            .map(|fp| found_map.remove(fp)) 
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
                "INSERT INTO fuzzy_cache (fingerprint, input_text, data) VALUES (?1, ?2, ?3)"
            )?;
            for (fp, text, data) in items {
                stmt.execute(params![fp as i64, text, data])?;
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
