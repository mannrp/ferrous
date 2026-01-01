use murmurhash3::murmurhash3_x64_128;

/// SimHash is a locality-sensitive hashing algorithm used to identify "near-duplicates" in text.
/// 
/// How it works:
/// 1. Tokenize text into shingles (overlapping n-grams).
/// 2. Hash each shingle into a 64-bit integer.
/// 3. For each hash, iterate over its 64 bits. If bit `i` is 1, increment weight `v[i]`. If 0, decrement `v[i]`.
/// 4. The final fingerprint bit `i` is 1 if `v[i]` > 0, else 0.
#[derive(Clone)]
pub struct SimHash {
    shingle_size: usize,
}

impl SimHash {
    pub fn new(shingle_size: usize) -> Self {
        Self { shingle_size }
    }

    /// Computes a 64-bit fingerprint for the given text.
    pub fn fingerprint(&self, text: &str) -> u64 {
        let mut weights = [0i32; 64];
        let shingles = self.get_shingles(text);

        for shingle in shingles {
            // We use MurmurHash3 for fast, high-quality 64-bit hashes.
            // murmurhash3_x64_128 returns (u64, u64). We take the first 64 bits.
            let (hash, _) = murmurhash3_x64_128(shingle.as_bytes(), 0);
            
            for i in 0..64 {
                let bit = (hash >> i) & 1;
                if bit == 1 {
                    weights[i] += 1;
                } else {
                    weights[i] -= 1;
                }
            }
        }

        let mut fingerprint: u64 = 0;
        for i in 0..64 {
            if weights[i] > 0 {
                fingerprint |= 1 << i;
            }
        }
        fingerprint
    }

    /// Calculates the Hamming Distance between two fingerprints.
    /// Hamming distance is the number of bits that differ.
    /// Lower distance = higher similarity.
    pub fn hamming_distance(a: u64, b: u64) -> u32 {
        (a ^ b).count_ones()
    }

    /// Internal helper to break text into overlapping n-grams (shingles).
    fn get_shingles(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < self.shingle_size {
            return vec![text.to_string()];
        }

        chars
            .windows(self.shingle_size)
            .map(|w| w.iter().collect::<String>())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simhash_similarity() {
        let hasher = SimHash::new(3);
        let f1 = hasher.fingerprint("The quick brown fox jumps over the lazy dog");
        let f2 = hasher.fingerprint("The quick brown fox jumps over the lazy dog!"); // One char diff
        let f3 = hasher.fingerprint("Something completely different in the world");

        let dist_near = SimHash::hamming_distance(f1, f2);
        let dist_far = SimHash::hamming_distance(f1, f3);

        // Key assertion: similar strings should have MUCH lower distance
        assert!(dist_near < dist_far, "Near strings should be closer than far strings");
        // Near strings should be reasonably close (shingle effects can cause >3 bit flips)
        assert!(dist_near <= 15, "Similar strings should have distance <= 15, got {}", dist_near);
    }
}
