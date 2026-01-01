use tiktoken_rs::CoreBPE;
use tokenizers::Tokenizer;

/// A trait for abstracting over different tokenizer backends (OpenAI vs HuggingFace).
/// This allows us to use the fastest implementation for the job.
pub trait TokenizerBackend: Send + Sync {
    /// Count tokens in text (Fast path for simple length checks)
    fn count(&self, text: &str) -> usize;

    /// Encode text into token IDs/Offsets for splitting
    /// Returns a vector of (start, end) byte offsets for each token.
    fn encode_offsets(&self, text: &str) -> Vec<(usize, usize)>;
}

/// Wrapper Enum to hold the concrete implementation
pub enum FerrousTokenizer {
    Tiktoken(CoreBPE),
    HuggingFace(Tokenizer),
}

impl TokenizerBackend for FerrousTokenizer {
    fn count(&self, text: &str) -> usize {
        match self {
            FerrousTokenizer::Tiktoken(bpe) => {
                // tiktoken is optimized for counting
                bpe.encode_with_special_tokens(text).len()
            }
            FerrousTokenizer::HuggingFace(tokenizer) => {
                tokenizer.encode(text, false)
                    .map(|e| e.len())
                    .unwrap_or(0)
            }
        }
    }

    fn encode_offsets(&self, text: &str) -> Vec<(usize, usize)> {
        match self {
            FerrousTokenizer::Tiktoken(_bpe) => {
                // tiktoken doesn't give offsets natively in exact same way, 
                // but we can simulate or use split_by_token logic.
                // For now, let's just get the tokens and mapping is tricky without offsets.
                // WARNING: tiktoken-rs standard API doesn't expose offsets easily.
                // We might need to implement a hack or just use it for counting if we don't strict split?
                // But Semantic Packing needs splits.
                
                // Correction: We really need offsets. If tiktoken can't give them fast, 
                // we might need to rely on `tokenizers` for everything or find a Tiktoken wrapper.
                // Actually `tiktoken-rs` exposes `encode_ordinary` which returns ids.
                // But to get offsets, we'd need to re-decode or track.
                
                // User mentioned: "tiktoken-rs might require a little work to map tokens back to byte ranges"
                // Let's implement a rudimentary offset tracker if needed, or fallback to slower path.
                // Ideally, we focus on `tokenizers` for structure and `tiktoken` for check?
                // But strictly speaking, the user wanted Dual Backend.
                
                // Let's leave a TODO and implement the HuggingFace one properly first.
                // Tiktoken offset mapping is non-trivial without re-decoding.
                vec![] // Placeholder
            }
            FerrousTokenizer::HuggingFace(tokenizer) => {
                let encoding = tokenizer.encode(text, false).unwrap_or_default();
                encoding.get_offsets().to_vec()
            }
        }
    }
}

impl FerrousTokenizer {
    pub fn new_from_name(name: &str) -> Option<Self> {
        // Try tiktoken first
        if let Ok(bpe) = tiktoken_rs::get_bpe_from_model(name) {
            return Some(FerrousTokenizer::Tiktoken(bpe));
        }
        None
    }

    pub fn new_from_file(path: &str) -> Option<Self> {
        if let Ok(tokenizer) = Tokenizer::from_file(path) {
            Some(FerrousTokenizer::HuggingFace(tokenizer))
        } else {
            None
        }
    }
}
