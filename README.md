# Ferrous

High-performance RAG primitives for Python, written in Rust.

Ferrous provides atomic, high-speed utilities designed to replace computational bottlenecks in modern Retrieval-Augmented Generation (RAG) pipelines. It focuses on zero-cost abstractions, minimal overhead, and systems-level performance for common text processing tasks.

## What's New in v0.3 (The Polyglot Update)

**Parallel Semantic Tokenization** – The `MarkdownChunker` now supports token-based splitting with pluggable tokenizer backends:

- **OpenAI Models**: Built-in support for GPT-3.5, GPT-4, and `cl100k_base` via `tiktoken-rs`
- **HuggingFace Models**: Load any `tokenizer.json` (Llama 3, Mistral, Gemma, etc.) via the `tokenizers` crate
- **Backward Compatible**: Legacy `max_characters` mode still works

This delivers **9x+ speedup** over LangChain's tokenized splitter while producing denser, more cost-efficient chunks.

## Key Primitives

### FuzzyCache
A lexical caching layer using SimHash (locality-sensitive hashing) to detect near-duplicate queries and content.
- **Use Case:** Avoid redundant embedding API calls for slightly modified or repetitive user queries.
- **Backend:** SQLite for persistent, serverless storage.
- **Performance:** Sub-millisecond fingerprinting and O(log N) lookup.

### MarkdownChunker
A structure-aware document splitter that leverages a formal Markdown AST parser.
- **Use Case:** Splitting documents while preserving the integrity of headers, paragraphs, and code blocks.
- **Accuracy:** Eliminates semantic breakage caused by naive character or token-based splitters.
- **v0.3**: Supports precise token-based splitting with OpenAI and HuggingFace tokenizers.

### ContextPacker
An importance-based context compression utility using the TextRank graph algorithm.
- **Use Case:** Ranking retrieved document segments and packing the most information-dense content into a fixed token budget.
- **Diversity:** Implements relevance-weighted selection to ensure context diversity and reduce redundancy.
- **Attention-Aware:** Optional U-shaped ordering places important content at context boundaries where LLMs attend most (Liu et al., 2023).

## Installation

```bash
pip install ferrous
```

## Quick Start

### Caching
```python
from ferrous import FuzzyCache

cache = FuzzyCache("cache.db", threshold=2)
if not cache.get(query):
    result = expensive_api_call(query)
    cache.put(query, result)
```

### Chunking (v0.3 Token-Based)
```python
from ferrous import MarkdownChunker

# NEW: Token-based chunking with OpenAI tokenizer
chunker = MarkdownChunker(tokenizer_name="gpt-4", max_tokens=512)
chunks = chunker.chunk(markdown_text)

# Or load a HuggingFace tokenizer from file
chunker = MarkdownChunker(tokenizer_path="./tokenizer.json", max_tokens=512)
chunks = chunker.chunk(markdown_text)

# Legacy: Character-based chunking still works
chunker = MarkdownChunker(max_characters=2000)
chunks = chunker.chunk(markdown_text)
```

### Packing
```python
from ferrous import ContextPacker

# Default: ranked by importance
packer = ContextPacker(max_chars=2048)
packed_context = packer.pack(document_list)

# U-shaped: important content at start and end of context
packer = ContextPacker(max_chars=2048, strategy="u_shaped")
packed_context = packer.pack(document_list)
```

## Performance Benchmarks

### v0.3 Tokenized Chunking (5MB Markdown)

| Implementation | Time (ms) | Chunks | Density (Tok/Chunk) | Speedup |
| :--- | ---: | ---: | ---: | ---: |
| LangChain (Tokenized) | 3,119 ms | 2,760 | 461 | 1x |
| **Ferrous v0.3 (Tokens)** | **336 ms** | **2,514** | **507** | **9.3x** |
| Ferrous v0.2 (Chars) | 28 ms | 2,539 | 503 | Reference |

Key takeaways:
- **9.3x faster** than LangChain's tokenized splitter
- **~10% fewer chunks** = lower retrieval costs
- **Higher density** = better token budget utilization

### Legacy Benchmarks (200KB+ payloads)

| Task | Implementation | Latency | Speedup |
| :--- | :--- | :--- | :--- |
| **Markdown Chunking** | LangChain (Python) | 81.25 ms | 1x |
| | **Ferrous (Rust)** | **0.95 ms** | **85.5x** |
| **Fuzzy Cache Lookup**| **Ferrous (SimHash)**| **0.34 ms** | **N/A** |
| **TextRank Packing** | **Ferrous (Rust)** | **35.96 ms / doc**| **N/A** |

*Note: Benchmarks performed on Windows 10. Performance may vary by hardware.*

## Tokenizer Support (v0.3)

| Tokenizer | Type | Example Usage |
| :--- | :--- | :--- |
| `gpt-4` | OpenAI | `tokenizer_name="gpt-4"` |
| `gpt-3.5-turbo` | OpenAI | `tokenizer_name="gpt-3.5-turbo"` |
| `text-embedding-ada-002` | OpenAI | `tokenizer_name="text-embedding-ada-002"` |
| Custom HuggingFace | File | `tokenizer_path="./llama3_tokenizer.json"` |

## Performance Note
Ferrous is built in Rust with PyO3 bindings. It aims for a 10x-100x performance improvement over standard Python implementations for text graph processing and structural parsing. It requires no GPU and has no heavy neural network dependencies.

## License
MIT