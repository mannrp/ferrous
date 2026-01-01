use std::collections::{HashMap, HashSet};

/// A lightweight TF-IDF scorer that calculates Term Frequency - Inverse Document Frequency
/// to identify "important" sentences within a large text.
/// 
/// It uses "Batch IDF" - meaning it calculates IDF statistics solely from the current
/// batch of documents being processed, effectively treating the batch as the entire corpus.
/// This works well for identifying sentences that contain terms rare *in this specific context*
/// but frequent in the sentence (discriminative terms).
pub struct TfidfScorer;

impl TfidfScorer {
    /// Scores sentences based on the sum of TF-IDF scores of their terms.
    /// Returns a list of (score, original_index) tuples, sorted by score descending.
    pub fn score_sentences(sentences: &[String]) -> Vec<(f64, usize)> {
        let n = sentences.len();
        if n == 0 { return vec![]; }

        // 1. Tokenize all sentences
        let tokenized: Vec<Vec<String>> = sentences.iter()
            .map(|s| Self::tokenize(s))
            .collect();

        // 2. Calculate Document Frequency (DF) for every term
        // Ideally, we'd have a global IDF, but "Batch IDF" works for relative importance.
        let mut df: HashMap<String, usize> = HashMap::new();
        for tokens in &tokenized {
            let unique_terms: HashSet<&String> = tokens.iter().collect();
            for term in unique_terms {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
        }

        // 3. Score each sentence
        // Score = Sum(TF * IDF) / log(Length)  <-- Length normalization is crucial
        let mut scores: Vec<(f64, usize)> = Vec::with_capacity(n);

        for (i, tokens) in tokenized.iter().enumerate() {
            if tokens.is_empty() {
                scores.push((0.0, i));
                continue;
            }

            let mut term_counts: HashMap<&String, usize> = HashMap::new();
            for t in tokens {
                *term_counts.entry(t).or_insert(0) += 1;
            }

            let mut score = 0.0;
            for (term, count) in &term_counts {
                // TF: raw count
                let tf = *count as f64;
                
                // IDF: ln(N / (1 + df))
                // Add 1 to DF to avoid division by zero (though unlikely here)
                let doc_freq = *df.get(*term).unwrap_or(&0) as f64;
                let idf = (n as f64 / (1.0 + doc_freq)).ln();
                
                score += tf * idf;
            }

            // Length Normalization
            // Longer sentences naturally have higher sums. We want dense information.
            // We normalize by log(len) rather than len to still slightly favor longer sentences (more context).
            let len_norm = (tokens.len() as f64).ln().max(1.0);
            
            scores.push((score / len_norm, i));
        }

        scores
    }

    /// Simple tokenizer: lowercase, alphanumerics only, ignore short stopwords
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() > 2) // Filter "is", "at", "to" etc.
            .map(|s| s.to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tfidf_prioritizes_rare_terms() {
        let sentences = vec![
            "The quick brown fox jumps.".to_string(),      // "fox", "jumps" (rare)
            "The the the the the.".to_string(),            // "the" (common, ignored by len filter)
            "Another generic sentence.".to_string(),
            "The quick brown fox jumps again.".to_string(), // Similar to first
        ];

        let scores = TfidfScorer::score_sentences(&sentences);
        println!("SCORES: {:?}", scores);
        
        // The sentences with "fox" and "jumps" should score highest
        // Sentence 1 ("The the...") has no valid tokens (>2 chars), so score 0
        
        // Sort explicitly by score to verify
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        println!("SORTED: {:?}", sorted);

        // Sentence 2 ("Another generic sentence") is all unique, rare words. It scores highest.
        // Sentence 3 ("...agains") has "again" (rare) plus "fox" (semi-rare).
        // Sentence 0 ("...jumps") is a subset of 3, so lower score.
        // Sentence 1 ("The...") is all stop words/common words. Score 0.
        
        println!("ORDER: {:?}", sorted.iter().map(|s| s.1).collect::<Vec<_>>());

        // We expect: [2, 3, 0, 1]
        assert_eq!(sorted[0].1, 2, "index 2 (Another...) should be #1");
        assert_eq!(sorted[1].1, 3, "index 3 (again) should be #2"); 
        assert_eq!(sorted[2].1, 0, "index 0 (base) should be #3");
        assert_eq!(sorted[3].1, 1, "index 1 (garbage) should be #4");
    }
}
