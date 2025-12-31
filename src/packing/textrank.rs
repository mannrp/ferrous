use std::collections::HashMap;

/// TextRank is a graph-based ranking algorithm for keyword and sentence extraction.
/// 
/// It treats sentences as nodes in a graph and uses similarity (LexRank/Cosine) 
/// to define edge weights. It then runs PageRank to find the most "central" sentences.
pub struct TextRank {
    pub damping: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for TextRank {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 100,
            tolerance: 1e-4,
        }
    }
}

impl TextRank {
    /// Ranks a list of sentences based on their mutual similarity.
    /// 
    /// Returns a list of (score, index) pairs, sorted by score descending.
    pub fn rank_sentences(&self, sentences: &[String]) -> Vec<(f64, usize)> {
        let n = sentences.len();
        if n == 0 { return vec![]; }
        if n == 1 { return vec![(1.0, 0)]; }

        // 1. Build Similarity Matrix (Edge weights)
        let mut adj = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = self.similarity(&sentences[i], &sentences[j]);
                adj[i][j] = sim;
                adj[j][i] = sim;
            }
        }

        // 2. Normalize adjacency matrix (PageRank requirement)
        let mut weights = vec![0.0; n];
        for i in 0..n {
            let row_sum: f64 = adj[i].iter().sum();
            if row_sum > 0.0 {
                for j in 0..n {
                    adj[i][j] /= row_sum;
                }
            }
            weights[i] = 1.0 / (n as f64); // Initial rank
        }

        // 3. Power Iteration (PageRank)
        for _ in 0..self.max_iterations {
            let mut next_weights = vec![0.0; n];
            let mut diff = 0.0;

            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += adj[j][i] * weights[j];
                }
                next_weights[i] = (1.0 - self.damping) / (n as f64) + self.damping * sum;
                diff += (next_weights[i] - weights[i]).abs();
            }

            weights = next_weights;
            if diff < self.tolerance {
                break;
            }
        }

        let mut results: Vec<(f64, usize)> = weights.into_iter().enumerate().map(|(i, w)| (w, i)).collect();
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        results
    }

    /// Simple lexical similarity based on shared tokens.
    /// In a more advanced version, we would use TF-IDF or Embeddings.
    fn similarity(&self, a: &str, b: &str) -> f64 {
        let words_a = self.get_tokens(a);
        let words_b = self.get_tokens(b);

        if words_a.is_empty() || words_b.is_empty() {
            return 0.0;
        }

        let intersection = words_a.keys().filter(|k| words_b.contains_key(*k)).count();
        // Log-based similarity to avoid bias towards extremely long sentences.
        (intersection as f64) / ((words_a.len() as f64).log10() + (words_b.len() as f64).log10() + 1.0)
    }

    fn get_tokens(&self, text: &str) -> HashMap<String, usize> {
        let mut map = HashMap::new();
        // Lowercase and split by non-alphanumeric characters
        for word in text.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
            if word.len() > 2 { // Ignore short stop-words
                *map.entry(word.to_string()).or_insert(0) += 1;
            }
        }
        map
    }
}
