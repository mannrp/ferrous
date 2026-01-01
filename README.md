# Ferrous

High-performance RAG primitives for Python, written in Rust.

Ferrous provides atomic, high-speed utilities designed to replace computational bottlenecks in modern Retrieval-Augmented Generation (RAG) pipelines. It focuses on zero-cost abstractions, minimal overhead, and systems-level performance for common text processing tasks.

## What's New in v0.3 (The Polyglot Update)

**Token-Accurate Splitting** – The `MarkdownChunker` now supports precise token counting via pluggable tokenizer backends:

- **OpenAI Models**: GPT-5, GPT-4, GPT-3.5 and embeddings via `tiktoken-rs`
- **HuggingFace Models**: Load any `tokenizer.json` (Llama 3, Mistral, Gemma, etc.)
- **Backward Compatible**: Fast character-based mode (`max_characters`) still available

**Why this matters:** Character-based chunking guesses token counts (~4 chars/token) with ~20-30% error. Token-based chunking knows exactly how many tokens fit in your context window—no truncation, no wasted space.

**U-Shaped Packing** – Places important content at the start and end of context windows, based on ["Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172) (Liu et al., 2023). LLMs attend more to context boundaries; this exploits that pattern.

## Key Primitives

### FuzzyCache
A lexical caching layer using SimHash (locality-sensitive hashing) to detect near-duplicate queries and content.
- **Use Case:** Avoid redundant embedding API calls for slightly modified or repetitive user queries.
- **Backend:** SQLite for persistent, serverless storage.
- **Performance:** Sub-millisecond fingerprinting and O(log N) lookup.

### MarkdownChunker
A structure-aware document splitter that leverages a formal Markdown AST parser.
- **Use Case:** Splitting documents while preserving the integrity of headers, paragraphs, and code blocks.
- **v0.3**: Precise token-based splitting with OpenAI and HuggingFace tokenizers.

### ContextPacker
An importance-based context compression utility using the TextRank graph algorithm.
- **Use Case:** Ranking retrieved document segments and packing the most information-dense content into a fixed token budget.
- **v0.3**: Optional U-shaped ordering for improved LLM attention.

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

### Chunking
```python
from ferrous import MarkdownChunker

# Token-accurate: knows exactly how many tokens per chunk
chunker = MarkdownChunker(tokenizer_name="gpt-4", max_tokens=512)
chunks = chunker.chunk(markdown_text)

# Or load a HuggingFace tokenizer
chunker = MarkdownChunker(tokenizer_path="./tokenizer.json", max_tokens=512)

# Fast mode: character-based (less accurate, 10x faster)
chunker = MarkdownChunker(max_characters=2000)
```

### Packing
```python
from ferrous import ContextPacker

# Default: ranked by importance
packer = ContextPacker(max_chars=2048)
packed_context = packer.pack(document_list)

# U-shaped: important content at start and end of context
packer = ContextPacker(max_chars=2048, strategy="u_shaped")
```

## Performance

### Tokenized Chunking (5MB Markdown)

| Mode | Time | Chunks | Accuracy |
| :--- | ---: | ---: | :--- |
| **Ferrous (Tokens)** | 336 ms | 2,514 | Exact token count |
| LangChain (Tokens) | 3,119 ms | 2,760 | Exact token count |
| Ferrous (Chars) | 28 ms | 2,539 | ~20-30% estimation error |

- **9x faster** than LangChain when you need token accuracy
- **10x faster** character mode when you don't

### Other Benchmarks

| Component | Ferrous | Alternative | Speedup |
| :--- | ---: | ---: | ---: |
| Markdown Chunking (200KB) | 0.95 ms | 81 ms (LangChain) | 85x |
| Cache Lookup | 0.11 ms | 50 ms (API call) | 450x |
| TextRank Packing | 165 ms | 771 ms (Python) | 4.7x |

## Tokenizer Support

Ferrous uses `tiktoken-rs` which supports OpenAI model names:

| Model | Encoding | Usage |
| :--- | :--- | :--- |
| GPT-5, GPT-5.1, GPT-5.2 | `o200k_base` | `tokenizer_name="gpt-5"` |
| GPT-4, GPT-4-Turbo | `cl100k_base` | `tokenizer_name="gpt-4"` |
| GPT-3.5-Turbo | `cl100k_base` | `tokenizer_name="gpt-3.5-turbo"` |
| Embeddings | `cl100k_base` | `tokenizer_name="text-embedding-ada-002"` |
| Custom | Any | `tokenizer_path="./tokenizer.json"` |

*Note: tiktoken-rs model support depends on the crate version. For latest models, use `tokenizer_path` with the official tokenizer file.*

## References

- Liu et al. (2023). ["Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172). *Transactions of the Association for Computational Linguistics*.

## License

MIT