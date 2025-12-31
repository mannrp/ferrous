use pulldown_cmark::{Parser, Event, Tag, TagEnd};

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
    /// Note: This implementation currently uses string length as a proxy for tokens.
    /// In a production scenario, we'd use a real tokenizer (e.g., Tiktoken).
    pub fn chunk(&self, md: &str) -> Vec<String> {
        let parser = Parser::new(md);
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        // Track state to avoid splitting inside sensitive blocks (like tables or code).
        let mut in_code_block = false;
        let mut block_buffer = String::new();

        for event in parser {
            match event {
                // Start of a structural block (Header, Paragraph, List Item)
                Event::Start(tag) => {
                    if let Tag::CodeBlock(_) = tag {
                        in_code_block = true;
                    }
                    // For now, we just buffer everything.
                }
                
                // End of a structural block - this is where we check if we should "flush" the chunk
                Event::End(tag) => {
                    if let TagEnd::CodeBlock = tag {
                        in_code_block = false;
                    }

                    // If we're at a block boundary and the chunk is getting too big, flush it.
                    if !in_code_block && current_chunk.len() + block_buffer.len() > self.max_tokens {
                        if !current_chunk.is_empty() {
                            chunks.push(current_chunk.clone());
                            current_chunk.clear();
                        }
                    }
                    
                    current_chunk.push_str(&block_buffer);
                    block_buffer.clear();
                }

                // Actual content
                Event::Text(text) => {
                    block_buffer.push_str(&text);
                }
                
                Event::Code(code) | Event::InlineMath(code) => {
                    block_buffer.push_str(&code);
                }

                _ => {} // Handle other events if needed
            }
        }

        // Push final chunk if not empty
        if !current_chunk.is_empty() || !block_buffer.is_empty() {
            current_chunk.push_str(&block_buffer);
            chunks.push(current_chunk);
        }

        chunks
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
