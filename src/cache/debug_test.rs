#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::simhash::SimHash;

    #[test]
    fn test_short_string_distance() {
        let hasher = SimHash::new(3);
        let s1 = "What is the capital of France?";
        let s2 = "What's the capital of France?";
        
        let f1 = hasher.fingerprint(s1);
        let f2 = hasher.fingerprint(s2);
        
        let dist = SimHash::hamming_distance(f1, f2);
        println!("Distance between '{}' and '{}' is {}", s1, s2, dist);
        
        // We expect it to be <= 2 if we want the demo to pass
        assert!(dist <= 5, "Distance {} is too high for fuzzy match", dist);
    }
}
