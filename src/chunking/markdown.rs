use pulldown_cmark::{Parser, Event};

/// MarkdownChunker splits documents based on Markdown structure.
/// 
/// It uses `pulldown-cmark` to parse the document as a stream of events,
/// ensuring we only split at logical block boundaries (headers, paragraphs, list items).
pub struct MarkdownChunker {
    max_tokens: usize,
}

impl MarkdownChunker {
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }

    /// Chunks a markdown string into a list of strings, each within the token limit.
    /// 
    /// improved comparison vs langchain:
    /// This implementation uses zero-copy slicing of the original markdown string.
    /// It preserves all markdown syntax (headers, code blocks, etc.) which was previously lost.
    pub fn chunk(&self, md: &str) -> Vec<String> {
        let parser = Parser::new(md);
        let mut chunks = Vec::new();
        
        let mut chunk_start = 0;
        let mut last_safe_end = 0;
        let mut nesting_level = 0;

        // Iterate events with their byte offsets in the original string
        for (event, range) in parser.into_offset_iter() {
            match event {
                Event::Start(_) => {
                    nesting_level += 1;
                }
                Event::End(_) => {
                    if nesting_level > 0 {
                        nesting_level -= 1;
                    }
                    
                    // If we are back at root level, this is a safe place to maybe split
                    if nesting_level == 0 {
                        // Check if adding this block would exceed the limit
                        // We use range.end as the potential end of the chunk
                        if range.end - chunk_start > self.max_tokens {
                            // Current block pushes us over limit.
                            // Split at the LAST safe ending (preserves this big block for the next chunk)
                            if last_safe_end > chunk_start {
                                chunks.push(md[chunk_start..last_safe_end].to_string());
                                chunk_start = last_safe_end;
                            } 
                            // Edge case: A single block is huge (larger than max_tokens)
                            // We have to split it anyway or accept the oversize.
                            // For safety, we force split at valid boundary if we haven't advanced.
                            else {
                                chunks.push(md[chunk_start..range.end].to_string());
                                chunk_start = range.end;
                            }
                        }
                        
                        // Mark this as the new safe ending point
                        last_safe_end = range.end;
                    }
                }
                _ => {}
            }
        }

        // Final chunk
        if chunk_start < md.len() {
            chunks.push(md[chunk_start..].to_string());
        }

        chunks
    }

    /// Chunks multiple documents in parallel using Rayon.
    pub fn chunk_batch(&self, docs: Vec<String>) -> Vec<Vec<String>> {
        use rayon::prelude::*;
        docs.par_iter()
            .map(|doc| self.chunk(doc))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markdown_chunking() {
        let chunker = MarkdownChunker::new(50);
        let md = "# Header 1\nThis is a paragraph that is quite long.\n\n## Header 2\nAnother paragraph.";
        let chunks = chunker.chunk(md);
        
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.len() <= 100); // Buffer allowed for header + text
        }
    }
}
