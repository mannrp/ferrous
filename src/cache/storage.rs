use rusqlite::{params, Connection, Result};
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
