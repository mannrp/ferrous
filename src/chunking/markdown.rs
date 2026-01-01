use pulldown_cmark::{Parser, Event};
use rayon::prelude::*;
use std::sync::Arc;
use crate::tokenization::TokenizerBackend;

/// MarkdownChunker splits documents based on Markdown structure.
pub struct MarkdownChunker {
    max_tokens: usize,
    tokenizer: Option<Arc<dyn TokenizerBackend>>,
}

impl MarkdownChunker {
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens, tokenizer: None }
    }

    pub fn with_tokenizer(max_tokens: usize, tokenizer: Arc<dyn TokenizerBackend>) -> Self {
        Self { max_tokens, tokenizer: Some(tokenizer) }
    }

    /// Chunks a markdown string.
    /// If a tokenizer is present, it uses precise token counting and parallel processing.
    /// If not, it falls back to fast character-based splitting.
    pub fn chunk(&self, md: &str) -> Vec<String> {
        if let Some(tokenizer) = &self.tokenizer {
            self.chunk_with_tokenizer(md, tokenizer)
        } else {
            self.chunk_legacy_char(md)
        }
    }

    fn chunk_with_tokenizer(&self, md: &str, tokenizer: &Arc<dyn TokenizerBackend>) -> Vec<String> {
        // Step 1: Identify top-level blocks (Structural Parsing)
        let blocks = self.get_top_level_blocks(md);

        // Step 2: Tokenize blocks in parallel (SOTA Speed)
        // We collect (range, token_count) pairs
        let scored_blocks: Vec<(std::ops::Range<usize>, usize)> = blocks.into_par_iter()
            .map(|range| {
                let text = &md[range.clone()];
                let count = tokenizer.count(text);
                (range, count)
            })
            .collect();

        // Step 3: Semantic Bin Packing (Greedy but boundary-aware)
        let mut chunks = Vec::new();
        let mut current_chunk_start = if scored_blocks.is_empty() { 0 } else { scored_blocks[0].0.start };
        let mut current_chunk_end = current_chunk_start;
        let mut current_tokens = 0;

        for (range, tokens) in scored_blocks {
            // Gap handling: If there was a gap between blocks (e.g., newlines), 
            // we should conceptually include it in the *current* chunk if extending.
            // But `get_top_level_blocks` might skip whitespace. 
            // A simple approach is to just use range.end as the new end.
            
            if current_tokens + tokens > self.max_tokens {
                if current_tokens > 0 {
                    // Push current chunk
                    // Extend to include any trailing whitespace before this new block starts? 
                    // Actually `md[current_chunk_start..current_chunk_end]` covers the blocks we added.
                    chunks.push(md[current_chunk_start..current_chunk_end].to_string());
                }
                
                // Start new chunk with this block
                // Handle edge case: Single block > max_tokens
                if tokens > self.max_tokens {
                    // We must split this block.
                    // For now, naive fallback: just accept it or warn.
                    // Ideally we recurse inside the block, but for v0.3 start we accept "Block Integrity".
                    chunks.push(md[range.start..range.end].to_string());
                    current_tokens = 0;
                    current_chunk_start = range.end; // technically next block start
                } else {
                    current_chunk_start = range.start;
                    current_chunk_end = range.end;
                    current_tokens = tokens;
                }
            } else {
                // Add to current chunk
                current_chunk_end = range.end;
                current_tokens += tokens;
            }
        }

        // Push final chunk
        if current_tokens > 0 {
             chunks.push(md[current_chunk_start..current_chunk_end].to_string());
        }

        chunks
    }

    fn get_top_level_blocks(&self, md: &str) -> Vec<std::ops::Range<usize>> {
        let parser = Parser::new(md);
        let mut blocks = Vec::new();
        let mut nesting_level = 0;
        let mut block_start = 0;

        for (event, range) in parser.into_offset_iter() {
            match event {
                Event::Start(_) => {
                    if nesting_level == 0 {
                        block_start = range.start;
                    }
                    nesting_level += 1;
                }
                Event::End(_) => {
                    if nesting_level > 0 {
                        nesting_level -= 1;
                    }
                    if nesting_level == 0 {
                        blocks.push(block_start..range.end);
                    }
                }
                _ => {}
            }
        }
        blocks
    }

    fn chunk_legacy_char(&self, md: &str) -> Vec<String> {
        let parser = Parser::new(md);
        let mut chunks = Vec::new();
        
        let mut chunk_start = 0;
        let mut last_safe_end = 0;
        let mut nesting_level = 0;

        for (event, range) in parser.into_offset_iter() {
            match event {
                Event::Start(_) => {
                    nesting_level += 1;
                }
                Event::End(_) => {
                    if nesting_level > 0 { nesting_level -= 1; }
                    
                    if nesting_level == 0 {
                        if range.end - chunk_start > self.max_tokens {
                            if last_safe_end > chunk_start {
                                chunks.push(md[chunk_start..last_safe_end].to_string());
                                chunk_start = last_safe_end;
                            } else {
                                chunks.push(md[chunk_start..range.end].to_string());
                                chunk_start = range.end;
                            }
                        }
                        last_safe_end = range.end;
                    }
                }
                _ => {}
            }
        }
        if chunk_start < md.len() {
            chunks.push(md[chunk_start..].to_string());
        }
        chunks
    }

    pub fn chunk_batch(&self, docs: Vec<String>) -> Vec<Vec<String>> {
        docs.par_iter()
            .map(|doc| self.chunk(doc))
            .collect()
    }
}
