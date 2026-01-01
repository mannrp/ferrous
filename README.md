# Ferrous

High-performance RAG primitives for Python, written in Rust.

Ferrous provides atomic, high-speed utilities designed to replace computational bottlenecks in modern Retrieval-Augmented Generation (RAG) pipelines. It focuses on zero-cost abstractions, minimal overhead, and systems-level performance for common text processing tasks.

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

### ContextPacker
An importance-based context compression utility using the TextRank graph algorithm.
- **Use Case:** Ranking retrieved document segments and packing the most information-dense content into a fixed token budget.
- **Diversity:** Implements relevance-weighted selection to ensure context diversity and reduce redundancy.

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

chunker = MarkdownChunker(max_tokens=512)
chunks = chunker.chunk(markdown_text)
```

### Packing
```python
from ferrous import ContextPacker

packer = ContextPacker(max_tokens=2048)
packed_context = packer.pack(document_list)
```

## Performance Benchmarks

The following benchmarks were conducted on 200KB+ payloads, comparing Ferrous to industry-standard Python implementations (LangChain).

| Task | Implementation | Latency | Speedup |
| :--- | :--- | :--- | :--- |
| **Markdown Chunking** | LangChain (Python) | 81.25 ms | 1x |
| | **Ferrous (Rust)** | **0.95 ms** | **85.5x** |
| **Fuzzy Cache Lookup**| **Ferrous (SimHash)**| **0.34 ms** | **N/A** |
| **TextRank Packing** | **Ferrous (Rust)** | **35.96 ms / doc**| **N/A** |

*Note: Benchmarks performed on Windows 10, AMD/Intel processors may vary. Fuzzy cache lookup includes SQLite overhead.*

## Performance Note
Ferrous is built in Rust with PyO3 bindings. It aims for a 10x-100x performance improvement over standard Python implementations for text graph processing and structural parsing. It requires no GPU and has no heavy neural network dependencies by default.

## License
MIT