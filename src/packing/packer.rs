use crate::packing::textrank::TextRank;
use unicode_segmentation::UnicodeSegmentation;

/// ContextPacker takes a large number of retrieved documents and "packs" them 
/// into a smaller token budget using importance-based ranking (TextRank) 
/// and diversity-based selection (MMR).
pub struct ContextPacker {
    max_chars: usize,
    ranker: TextRank,
}

impl ContextPacker {
    pub fn new(max_chars: usize) -> Self {
        Self {
            max_chars,
            ranker: TextRank::default(),
        }
    }

    /// Packs multiple documents into a single dense context string.
    pub fn pack(&self, documents: &[String]) -> String {
        // 1. Break documents into sentences using Unicode-compliant segmentation (UAX #29)
        // This correctly handles numbers ($3.50), URLs (example.com), and CJK text.
        let sentences: Vec<String> = documents.iter()
            .flat_map(|doc| doc.unicode_sentences())
            .map(|s| s.trim().to_string())
            .filter(|s| s.len() > 5)  // Skip very short fragments
            .collect();

        if sentences.is_empty() { return String::new(); }

        // 2. Cap sentence count to bound TextRank's O(n²) complexity
        // 300 sentences = 90k comparisons, ~30ms.
        // If we have more than 300, we use TF-IDF (O(n)) to find the most "information-dense" 300.
        const MAX_SENTENCES: usize = 300;
        
        // This vector holds the INDICES of sentences we decide to keep
        let mut keep_indices: Vec<usize> = (0..sentences.len()).collect();

        if sentences.len() > MAX_SENTENCES {
            // Use TF-IDF to score all sentences
            let scores = crate::packing::tfidf::TfidfScorer::score_sentences(&sentences);
            
            // Sort by score descending
            let mut sorted_scores = scores;
            sorted_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            
            // Keep top MAX_SENTENCES
            keep_indices = sorted_scores.iter()
                .take(MAX_SENTENCES)
                .map(|(_, idx)| *idx)
                .collect();
                
            // Sort indices to maintain original document flow/order
            keep_indices.sort();
        }

        // Filter sentences to just the kept ones
        // We do this so TextRank only sees the "important" ones
        let filtered_sentences: Vec<String> = keep_indices.iter()
            .map(|&idx| sentences[idx].clone())
            .collect();

        // 3. Rank sentences using TextRank (on the filtered subset)
        let ranked = self.ranker.rank_sentences(&filtered_sentences);

        // 4. Selection (MMR - Simplified for V1)
        // We pick top sentences until the budget is full.
        // We skip sentences that are too similar to already selected ones.
        let mut selected = Vec::new();
        let mut current_chars = 0;

        for (_score, idx) in ranked {
            let sentence = &filtered_sentences[idx];
            let sentence_len = sentence.len(); 

            if current_chars + sentence_len <= self.max_chars {
                // Check if redundant (simple check)
                let is_redundant = selected.iter().any(|s: &String| {
                    self.is_duplicate(s, sentence)
                });

                if !is_redundant {
                    selected.push(sentence.clone());
                    current_chars += sentence_len;
                }
            }
        }

        selected.join("\n")
    }

    /// Packs multiple batches of documents in parallel using Rayon.
    pub fn pack_batch(&self, document_sets: Vec<Vec<String>>) -> Vec<String> {
        use rayon::prelude::*;
        document_sets.par_iter()
            .map(|docs| self.pack(docs))
            .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to extract sentences for testing segmentation behavior
    fn extract_sentences(text: &str) -> Vec<String> {
        text.unicode_sentences()
            .map(|s| s.trim().to_string())
            .filter(|s| s.len() > 5)
            .collect()
    }

    #[test]
    fn test_decimal_numbers_preserved() {
        // Decimal numbers should not cause sentence breaks
        // This is a key improvement over naive `.split('.')` 
        let text = "The price was $3.50 for the item. That's expensive.";
        let sentences = extract_sentences(text);
        
        // Should be 2 sentences, and the decimal should be intact
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].contains("$3.50"));
    }

    #[test]
    fn test_url_preserved() {
        // URLs should not be split on dots - this is where unicode-segmentation shines
        let text = "Visit example.com for more info. It has great content.";
        let sentences = extract_sentences(text);
        
        // Should be 2 sentences with URL intact
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].contains("example.com"));
    }

    #[test]
    fn test_basic_sentence_split() {
        // Normal sentences should split correctly on periods, !, and ?
        let text = "This is sentence one. This is sentence two! Is this sentence three?";
        let sentences = extract_sentences(text);
        
        assert_eq!(sentences.len(), 3);
    }
    
    #[test]
    fn test_ellipsis_handling() {
        // Ellipsis should not create multiple sentence breaks
        let text = "He paused... then continued speaking. The end.";
        let sentences = extract_sentences(text);
        
        // Should handle ellipsis reasonably (2 sentences)
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn test_packer_produces_output() {
        // End-to-end test: packer should produce meaningful output
        let packer = ContextPacker::new(1000);
        let docs = vec![
            "The quick brown fox jumps over the lazy dog. This is a classic sentence.".to_string(),
            "Another document with useful information. It contains multiple sentences.".to_string(),
        ];
        
        let result = packer.pack(&docs);
        
        // Result should contain properly segmented content
        assert!(!result.is_empty());
        // Should have at least some content
        assert!(result.len() > 20);
    }
    
    #[test]
    fn test_short_fragments_filtered() {
        // Very short fragments (<=5 chars) should be filtered out
        let text = "OK. This is a longer sentence here. Yes.";
        let sentences = extract_sentences(text);
        
        // "OK." and "Yes." are 2-3 chars after trim, should be filtered
        // Only the long sentence should remain
        assert_eq!(sentences.len(), 1);
        assert!(sentences[0].contains("longer sentence"));
    }
}
