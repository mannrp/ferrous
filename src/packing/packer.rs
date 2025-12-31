use crate::packing::textrank::TextRank;

/// ContextPacker takes a large number of retrieved documents and "packs" them 
/// into a smaller token budget using importance-based ranking (TextRank) 
/// and diversity-based selection (MMR).
pub struct ContextPacker {
    max_tokens: usize,
    ranker: TextRank,
}

impl ContextPacker {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            ranker: TextRank::default(),
        }
    }

    /// Packs multiple documents into a single dense context string.
    pub fn pack(&self, documents: &[String]) -> String {
        // 1. Break documents into sentences
        // Simple sentence splitting for V1
        let sentences: Vec<String> = documents.iter()
            .flat_map(|doc| doc.split_inclusive(&['.', '!', '?'][..]))
            .map(|s| s.trim().to_string())
            .filter(|s| s.len() > 5)
            .collect();

        if sentences.is_empty() { return String::new(); }

        // 2. Rank sentences using TextRank
        let ranked = self.ranker.rank_sentences(&sentences);

        // 3. Selection (MMR - Simplified for V1)
        // We pick top sentences until the budget is full.
        // We skip sentences that are too similar to already selected ones.
        let mut selected = Vec::new();
        let mut current_tokens = 0;

        for (_score, idx) in ranked {
            let sentence = &sentences[idx];
            let sentence_len = sentence.len(); // Proxy for tokens

            if current_tokens + sentence_len <= self.max_tokens {
                // Check if redundant (simple check)
                let is_redundant = selected.iter().any(|s: &String| {
                    self.is_duplicate(s, sentence)
                });

                if !is_redundant {
                    selected.push(sentence.clone());
                    current_tokens += sentence_len;
                }
            }
        }

        selected.join("\n")
    }

    /// Very simple Jaccard-like check for redundancy.
    fn is_duplicate(&self, a: &str, b: &str) -> bool {
        let set_a: std::collections::HashSet<_> = a.split_whitespace().collect();
        let set_b: std::collections::HashSet<_> = b.split_whitespace().collect();
        let common = set_a.intersection(&set_b).count();
        let overlap = (common as f64) / (set_a.len().min(set_b.len()) as f64);
        overlap > 0.8 // 80% word overlap is considered redundant
    }
}
