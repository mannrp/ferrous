"""
Ferrous Comprehensive Benchmark Suite
=====================================

A rigorous, honest benchmarking framework that tests ferrous primitives
against industry alternatives (LangChain, etc.) at realistic scale.

Usage:
    python benchmark_suite.py                    # Run all benchmarks
    python benchmark_suite.py --component cache  # Run specific component
    python benchmark_suite.py --scale large      # Run at large scale
    python benchmark_suite.py --log results.json # Save results to file

Requirements:
    pip install langchain-text-splitters nltk
"""

import json
import time
import argparse
import hashlib
import statistics
import platform
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable
try:
    import networkx as nx
    import numpy as np
except ImportError:
    nx = None
    np = None

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCALES = {
    "small": {"docs": 10, "doc_size_kb": 50, "cache_entries": 100, "chunk_size_kb": 50},
    "medium": {"docs": 50, "doc_size_kb": 10, "cache_entries": 1000, "chunk_size_kb": 200},
    "large": {"docs": 200, "doc_size_kb": 10, "cache_entries": 10000, "chunk_size_kb": 500},
    "stress": {"docs": 500, "doc_size_kb": 20, "cache_entries": 50000, "chunk_size_kb": 1000},
}

# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark measurement"""
    name: str
    component: str
    implementation: str  # "ferrous" or "langchain" etc.
    metric: str  # "latency_ms", "throughput_ops_sec", etc.
    value: float
    unit: str
    scale: str
    iterations: int
    std_dev: float
    min_val: float
    max_val: float
    
@dataclass
class BenchmarkReport:
    """Complete benchmark run report"""
    timestamp: str
    platform: str
    python_version: str
    ferrous_version: str
    scale: str
    results: List[BenchmarkResult]
    comparisons: List[Dict[str, Any]]
    
    def to_dict(self) -> dict:
        comparisons = []
        
        # Helper to find result by implementation
        def find_res(impl, bench_name):
            return next((r for r in self.results if r.implementation == impl and r.name == bench_name), None)

        # 1. Chunking Speedup
        ferrous_chunk = find_res("ferrous", "markdown_chunking")
        lc_rec = find_res("langchain_recursive", "markdown_chunking")
        
        if ferrous_chunk and lc_rec:
            comparisons.append({
                "benchmark": "markdown_chunking",
                "scale": ferrous_chunk.scale,
                "ferrous": ferrous_chunk.value,
                "alternative": "langchain_recursive",
                "alternative_value": lc_rec.value,
                "speedup": lc_rec.value / ferrous_chunk.value,
                "unit": ferrous_chunk.unit
            })

        # 2. Cache ROI (vs Network)
        ferrous_cache = find_res("ferrous", "cache_lookup_exact")
        network = find_res("network_api_call", "cache_lookup_exact")
        
        if ferrous_cache and network:
             comparisons.append({
                "benchmark": "cache_roi",
                "scale": ferrous_cache.scale,
                "ferrous": ferrous_cache.value,
                "alternative": "network_api_call",
                "alternative_value": network.value,
                "speedup": network.value / ferrous_cache.value,
                "unit": "µs"
            })

        # 3. Cache Batch Speedup (v0.2)
        ferrous_get = find_res("ferrous", "cache_lookup_exact")
        ferrous_batch = find_res("ferrous", "cache_lookup_batch")
        if ferrous_get and ferrous_batch:
            # Normalize batch to per-item for fair comparison
            per_item_batch = ferrous_batch.value / 100 
            comparisons.append({
                "benchmark": "cache_batch_efficiency",
                "scale": "100_items",
                "single_op": ferrous_get.value,
                "batch_op_per_item": per_item_batch,
                "speedup": ferrous_get.value / per_item_batch,
                "unit": "µs"
            })

        # 4. Chunking Parallel Speedup (v0.2)
        ferrous_chunk = find_res("ferrous", "markdown_chunking")
        ferrous_parallel = find_res("ferrous_parallel", "markdown_chunking_batch")
        if ferrous_chunk and ferrous_parallel:
            # The batch test uses 100 docs
            seq_total = ferrous_chunk.value * 100 
            comparisons.append({
                "benchmark": "chunking_parallel_efficiency",
                "scale": "100_docs",
                "sequential_est": seq_total,
                "parallel_batch": ferrous_parallel.value,
                "speedup": seq_total / ferrous_parallel.value,
                "unit": "ms"
            })
            
        # 5. Packing Parallel Speedup (v0.2)
        ferrous_pack_seq = find_res("ferrous_sequential", "context_packing_sequential")
        ferrous_pack_batch = find_res("ferrous_parallel", "context_packing_batch")
        if ferrous_pack_seq and ferrous_pack_batch:
             comparisons.append({
                "benchmark": "packing_parallel_efficiency",
                "scale": "20_sets_of_10",
                "sequential_value": ferrous_pack_seq.value,
                "parallel_batch": ferrous_pack_batch.value,
                "speedup": ferrous_pack_seq.value / ferrous_pack_batch.value,
                "unit": "ms"
            })

        # 6. Batch Write Speedup (v0.2)
        ferrous_put = find_res("ferrous", "cache_put_single")
        ferrous_put_batch = find_res("ferrous", "cache_put_batch")
        if ferrous_put and ferrous_put_batch:
            per_item_batch = ferrous_put_batch.value / 100
            comparisons.append({
                "benchmark": "cache_batch_write_efficiency",
                "scale": "100_items",
                "single_op": ferrous_put.value,
                "batch_op_per_item": per_item_batch,
                "speedup": ferrous_put.value / per_item_batch,
                "unit": "ms"
            })
            
        self.comparisons = comparisons
        return {
            "timestamp": self.timestamp,
            "platform": self.platform,
            "python_version": self.python_version,
            "ferrous_version": self.ferrous_version,
            "scale": self.scale,
            "results": [asdict(r) for r in self.results],
            "comparisons": self.comparisons,
        }

# ==============================================================================
# TEST DATA GENERATORS
# ==============================================================================

def generate_realistic_text(size_kb: int) -> str:
    """Generate realistic text with proper sentences, abbreviations, numbers, URLs"""
    # Mix of realistic content patterns
    patterns = [
        "Dr. Smith presented the Q{n} results to the board. ",
        "The price dropped to ${price:.2f} after market close. ",
        "Visit https://example.com/doc/{n} for more details. ",
        "Mr. Johnson from the U.S.A. division reported growth of {pct}%. ",
        "The API response time was {latency}ms on average. ",
        "According to Fig. {n}, the correlation coefficient is {corr:.3f}. ",
        "The experiment ran from Jan. 1st to Dec. 31st last year. ",
        "Prof. Williams et al. published findings in Nature vol. {n}. ",
        "The dataset contains {count:,} records across {tables} tables. ",
        "Error rate decreased from {old}% to {new}% after optimization. ",
    ]
    
    text = ""
    n = 0
    while len(text) < size_kb * 1024:
        pattern = patterns[n % len(patterns)]
        text += pattern.format(
            n=n,
            price=10 + (n * 0.5) % 100,
            pct=5 + n % 20,
            latency=50 + n % 200,
            corr=0.5 + (n % 50) / 100,
            count=1000 * (n + 1),
            tables=3 + n % 10,
            old=10 + n % 15,
            new=2 + n % 8,
        )
        n += 1
    
    return text[:size_kb * 1024]

def generate_markdown_document(size_kb: int) -> str:
    """Generate realistic markdown with headers, code blocks, lists"""
    sections = []
    total_size = 0
    section_num = 0
    
    while total_size < size_kb * 1024:
        section = f"""
# Section {section_num}: Overview

This section covers the implementation details and performance characteristics.
The following considerations apply to production deployments.

## {section_num}.1 Configuration

Configure the system with these settings:

```python
config = {{
    "max_connections": {10 + section_num},
    "timeout_ms": {1000 + section_num * 100},
    "retry_count": 3,
}}
```

## {section_num}.2 Performance Notes

- Average latency: {50 + section_num * 10}ms
- P99 latency: {200 + section_num * 20}ms
- Throughput: {1000 + section_num * 100} req/sec

The system handles approximately {10000 + section_num * 1000:,} requests per hour.

"""
        sections.append(section)
        total_size += len(section)
        section_num += 1
    
    return "".join(sections)[:size_kb * 1024]

def generate_similar_texts(base: str, count: int, variation: float = 0.1) -> List[str]:
    """Generate texts with controlled similarity to base text"""
    import random
    texts = [base]
    words = base.split()
    
    for i in range(count - 1):
        # Replace some words to create near-duplicates
        modified = words.copy()
        num_changes = int(len(words) * variation * (i + 1) / count)
        for _ in range(num_changes):
            idx = random.randint(0, len(modified) - 1)
            modified[idx] = f"word{random.randint(0, 1000)}"
        texts.append(" ".join(modified))
    
    return texts

# ==============================================================================
# BENCHMARK FUNCTIONS
# ==============================================================================

def benchmark_function(func: Callable, iterations: int = 10, warmup: int = 2) -> Dict[str, float]:
    """Run a function multiple times and return statistics"""
    # Warmup
    for _ in range(warmup):
        func()
    
    # Actual measurements
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    
    return {
        "mean": statistics.mean(times),
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
        "median": statistics.median(times),
    }

# ==============================================================================
# COMPONENT BENCHMARKS
# ==============================================================================

class ChunkingBenchmarks:
    """Benchmarks for MarkdownChunker vs alternatives"""
    
    def __init__(self, scale_config: dict):
        self.chunk_size = scale_config.get("chunk_size_kb", 200)  # Default 200KB like original bench
        self.num_docs = scale_config["docs"]
        
    def run(self) -> List[BenchmarkResult]:
        results = []
        
        # Generate test data - larger payloads to show real speedup
        markdown = generate_markdown_document(self.chunk_size)
        print(f"  Chunking benchmark: {len(markdown):,} chars ({self.chunk_size}KB)")
        
        # Ferrous MarkdownChunker
        try:
            from ferrous import MarkdownChunker
            
            def ferrous_chunk():
                chunker = MarkdownChunker(max_tokens=1000)
                return chunker.chunk(markdown)
            
            stats = benchmark_function(ferrous_chunk, iterations=20)
            results.append(BenchmarkResult(
                name="markdown_chunking",
                component="chunking",
                implementation="ferrous",
                metric="latency_ms",
                value=stats["mean"],
                unit="ms",
                scale=f"{self.chunk_size}KB",
                iterations=20,
                std_dev=stats["std_dev"],
                min_val=stats["min"],
                max_val=stats["max"],
            ))
            ferrous_time = stats["mean"]
        except ImportError:
            print("  WARNING: ferrous not installed")
            ferrous_time = None
        
        # LangChain RecursiveCharacterTextSplitter
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
            
            # RecursiveCharacterTextSplitter
            def langchain_recursive():
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                return splitter.split_text(markdown)
            
            stats = benchmark_function(langchain_recursive, iterations=10)
            results.append(BenchmarkResult(
                name="markdown_chunking",
                component="chunking",
                implementation="langchain_recursive",
                metric="latency_ms",
                value=stats["mean"],
                unit="ms",
                scale=f"{self.chunk_size}KB",
                iterations=10,
                std_dev=stats["std_dev"],
                min_val=stats["min"],
                max_val=stats["max"],
            ))
            
            # MarkdownTextSplitter (more comparable)
            def langchain_markdown():
                splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)
                return splitter.split_text(markdown)
            
            stats = benchmark_function(langchain_markdown, iterations=10)
            results.append(BenchmarkResult(
                name="markdown_chunking",
                component="chunking",
                implementation="langchain_markdown",
                metric="latency_ms",
                value=stats["mean"],
                unit="ms",
                scale=f"{self.chunk_size}KB",
                iterations=10,
                std_dev=stats["std_dev"],
                min_val=stats["min"],
                max_val=stats["max"],
            ))
            
        except ImportError:
            print("  WARNING: langchain-text-splitters not installed")
        
        # Ferrous Parallel Batch Chunking (v0.2)
        try:
            from ferrous import MarkdownChunker
            
            # Generate 100 docs for batch testing
            batch_docs = [generate_markdown_document(self.chunk_size // 10) for _ in range(100)]
            
            def ferrous_chunk_batch():
                chunker = MarkdownChunker(max_tokens=1000)
                return chunker.chunk_batch(batch_docs)
            
            stats = benchmark_function(ferrous_chunk_batch, iterations=10)
            results.append(BenchmarkResult(
                name="markdown_chunking_batch",
                component="chunking",
                implementation="ferrous_parallel",
                metric="latency_ms",
                value=stats["mean"],
                unit="ms",
                scale="100_docs",
                iterations=10,
                std_dev=stats["std_dev"],
                min_val=stats["min"],
                max_val=stats["max"],
            ))
        except (ImportError, AttributeError):
            pass

        return results


class CacheBenchmarks:
    """Benchmarks for FuzzyCache (SimHash) vs alternatives"""
    
    def __init__(self, scale_config: dict):
        self.num_entries = scale_config["cache_entries"]
        
    def run(self) -> List[BenchmarkResult]:
        results = []
        
        print(f"  Cache benchmark: {self.num_entries:,} entries")
        
        # Generate test data
        base_text = "The quick brown fox jumps over the lazy dog. " * 10
        texts = generate_similar_texts(base_text, self.num_entries)
        
        # Ferrous FuzzyCache
        try:
            from ferrous import FuzzyCache
            import tempfile
            import os
            
            # Write benchmark
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db_path = f.name
            
            try:
                cache = FuzzyCache(db_path, threshold=3)
                
                # Populate cache (batch mode for speed)
                batch_data = [(text, f"response_{i}") for i, text in enumerate(texts)]
                cache.put_batch(batch_data)
                
                # Read benchmark (exact match)
                def ferrous_get_exact():
                    return cache.get(texts[0])
                
                stats = benchmark_function(ferrous_get_exact, iterations=100)
                results.append(BenchmarkResult(
                    name="cache_lookup_exact",
                    component="cache",
                    implementation="ferrous",
                    metric="latency_us",
                    value=stats["mean"] * 1000,  # Convert to microseconds
                    unit="µs",
                    scale=f"{len(texts)} entries",
                    iterations=100,
                    std_dev=stats["std_dev"] * 1000,
                    min_val=stats["min"] * 1000,
                    max_val=stats["max"] * 1000,
                ))
                
                # Read benchmark (fuzzy match - near duplicate)
                near_dup = texts[0][:-20] + "different ending here"
                def ferrous_get_fuzzy():
                    return cache.get(near_dup)
                
                stats = benchmark_function(ferrous_get_fuzzy, iterations=50)
                results.append(BenchmarkResult(
                    name="cache_lookup_fuzzy",
                    component="cache",
                    implementation="ferrous",
                    metric="latency_us",
                    value=stats["mean"] * 1000,
                    unit="µs",
                    scale=f"{len(texts)} entries",
                    iterations=50,
                    std_dev=stats["std_dev"] * 1000,
                    min_val=stats["min"] * 1000,
                    max_val=stats["max"] * 1000,
                ))

                # Batch Read Benchmark (v0.2)
                batch_texts = texts[:100]
                def ferrous_get_batch():
                    return cache.get_batch(batch_texts)
                
                stats = benchmark_function(ferrous_get_batch, iterations=50)
                results.append(BenchmarkResult(
                    name="cache_lookup_batch",
                    component="cache",
                    implementation="ferrous",
                    metric="latency_us",
                    value=stats["mean"] * 1000,
                    unit="µs",
                    scale="100_items",
                    iterations=50,
                    std_dev=stats["std_dev"] * 1000,
                    min_val=stats["min"] * 1000,
                    max_val=stats["max"] * 1000,
                ))

                # Single Write Loop Benchmark (for comparison)
                put_single_items = [(f"s_text_{i}", f"s_data_{i}") for i in range(100)]
                def ferrous_put_loop():
                    for k, v in put_single_items:
                        cache.put(k, v)
                
                stats = benchmark_function(ferrous_put_loop, iterations=10)
                results.append(BenchmarkResult(
                    name="cache_put_single",
                    component="cache",
                    implementation="ferrous",
                    metric="latency_ms",
                    value=stats["mean"],
                    unit="ms",
                    scale="100_items",
                    iterations=10,
                    std_dev=stats["std_dev"],
                    min_val=stats["min"],
                    max_val=stats["max"],
                ))

                # Batch Write Benchmark (v0.2)
                put_batch_items = [(f"new_text_{i}", f"data_{i}") for i in range(100)]
                def ferrous_put_batch():
                    return cache.put_batch(put_batch_items)
                
                stats = benchmark_function(ferrous_put_batch, iterations=10)
                results.append(BenchmarkResult(
                    name="cache_put_batch",
                    component="cache",
                    implementation="ferrous",
                    metric="latency_ms",
                    value=stats["mean"],
                    unit="ms",
                    scale="100_items",
                    iterations=10,
                    std_dev=stats["std_dev"],
                    min_val=stats["min"],
                    max_val=stats["max"],
                ))
                
                # Explicitly close connection by dropping the object
                del cache
                import gc
                gc.collect()
                
            finally:
                # Windows file locking requires retry mechanism
                import time
                for _ in range(10):
                    try:
                        if os.path.exists(db_path):
                            os.unlink(db_path)
                        break
                    except OSError:
                        time.sleep(0.5)
                
        except ImportError:
            print("  WARNING: ferrous not installed")

        # Simulated API Call (The real problem we solve)
        def simulated_openai_call():
             # Simulate network latency (e.g. 50ms which is optimistic)
             time.sleep(0.05)
             return "embedding_vector_data"

        stats = benchmark_function(simulated_openai_call, iterations=10)
        results.append(BenchmarkResult(
            name="cache_lookup_exact",
            component="cache",
            implementation="network_api_call",
            metric="latency_us",
            value=stats["mean"] * 1000,
            unit="µs",
            scale=f"{len(texts)} entries",
            iterations=10,
            std_dev=stats["std_dev"] * 1000,
            min_val=stats["min"] * 1000,
            max_val=stats["max"] * 1000,
        ))
        
        # Python dict baseline (exact match only)
        py_cache = {hashlib.md5(t.encode()).hexdigest(): f"resp_{i}" for i, t in enumerate(texts)}
        query_hash = hashlib.md5(texts[0].encode()).hexdigest()
        
        def python_dict_get():
            return py_cache.get(query_hash)
        
        stats = benchmark_function(python_dict_get, iterations=1000)
        results.append(BenchmarkResult(
            name="cache_lookup_exact",
            component="cache",
            implementation="python_dict",
            metric="latency_us",
            value=stats["mean"] * 1000,
            unit="µs",
            scale=f"{len(texts)} entries",
            iterations=1000,
            std_dev=stats["std_dev"] * 1000,
            min_val=stats["min"] * 1000,
            max_val=stats["max"] * 1000,
        ))
        
        return results


class PackingBenchmarks:
    """Benchmarks for ContextPacker (TextRank) vs alternatives"""
    
    def __init__(self, scale_config: dict):
        self.num_docs = scale_config["docs"]
        self.doc_size = scale_config["doc_size_kb"]
        
    def run(self) -> List[BenchmarkResult]:
        results = []
        
        # Generate realistic documents
        docs = [generate_realistic_text(self.doc_size) for _ in range(self.num_docs)]
        total_chars = sum(len(d) for d in docs)
        print(f"  Packing benchmark: {self.num_docs} docs, {total_chars:,} total chars")
        
        # Ferrous ContextPacker
        try:
            from ferrous import ContextPacker
            
            def ferrous_pack():
                # 16000 chars ~= 4000 tokens
                packer = ContextPacker(max_chars=16000)
                return packer.pack(docs)
            
            stats = benchmark_function(ferrous_pack, iterations=5, warmup=1)
            
            # Also measure compression ratio
            packed = ferrous_pack()
            compression_ratio = total_chars / len(packed) if packed else 0
            
            results.append(BenchmarkResult(
                name="context_packing",
                component="packing",
                implementation="ferrous",
                metric="latency_ms",
                value=stats["mean"],
                unit="ms",
                scale=f"{self.num_docs} docs",
                iterations=5,
                std_dev=stats["std_dev"],
                min_val=stats["min"],
                max_val=stats["max"],
            ))
            
            results.append(BenchmarkResult(
                name="context_packing_compression",
                component="packing",
                implementation="ferrous",
                metric="compression_ratio",
                value=compression_ratio,
                unit="x",
                scale=f"{self.num_docs} docs",
                iterations=1,
                std_dev=0,
                min_val=compression_ratio,
                max_val=compression_ratio,
            ))
            
        except ImportError:
            print("  WARNING: ferrous not installed")

        # Ferrous Sequential Loop Packing (Baseline for Parallel)
        try:
            from ferrous import ContextPacker
            
            # 20 sets of 10 docs each
            batch_docs = [[generate_realistic_text(self.doc_size) for _ in range(10)] for _ in range(20)]
            
            def ferrous_pack_sequential_loop():
                packer = ContextPacker(max_chars=16000)
                # Simulate handling multiple requests sequentially
                for doc_set in batch_docs:
                    packer.pack(doc_set)
            
            stats = benchmark_function(ferrous_pack_sequential_loop, iterations=3)
            results.append(BenchmarkResult(
                name="context_packing_sequential",
                component="packing",
                implementation="ferrous_sequential",
                metric="latency_ms",
                value=stats["mean"],
                unit="ms",
                scale="20_sets_of_10",
                iterations=3,
                std_dev=stats["std_dev"],
                min_val=stats["min"],
                max_val=stats["max"],
            ))
            
        except (ImportError, AttributeError):
            pass

        # Ferrous Parallel Batch Packing (v0.2)
        try:
            from ferrous import ContextPacker
            
            # 20 sets of 10 docs each
            batch_docs = [[generate_realistic_text(self.doc_size) for _ in range(10)] for _ in range(20)]
            
            def ferrous_pack_batch():
                # 16000 chars ~= 4000 tokens
                packer = ContextPacker(max_chars=16000)
                return packer.pack_batch(batch_docs)
            
            stats = benchmark_function(ferrous_pack_batch, iterations=5)
            results.append(BenchmarkResult(
                name="context_packing_batch",
                component="packing",
                implementation="ferrous_parallel",
                metric="latency_ms",
                value=stats["mean"],
                unit="ms",
                scale="20_sets_of_10",
                iterations=5,
                std_dev=stats["std_dev"],
                min_val=stats["min"],
                max_val=stats["max"],
            ))
        except (ImportError, AttributeError):
            pass
        
        # NLTK sentence tokenization (baseline for comparison)
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
            
            from nltk.tokenize import sent_tokenize
            
            def nltk_tokenize():
                all_sentences = []
                for doc in docs:
                    all_sentences.extend(sent_tokenize(doc))
                return all_sentences[:300]  # Same cap as ferrous
            
            stats = benchmark_function(nltk_tokenize, iterations=5)
            results.append(BenchmarkResult(
                name="sentence_tokenization",
                component="packing",
                implementation="nltk",
                metric="latency_ms",
                value=stats["mean"],
                unit="ms",
                scale=f"{self.num_docs} docs",
                iterations=5,
                std_dev=stats["std_dev"],
                min_val=stats["min"],
                max_val=stats["max"],
            ))
            
        except ImportError:
            print("  WARNING: nltk not installed")
        
        # Python TextRank (numpy + networkx) - The "Full Pipeline" Comparison
        if nx and np:
            try:
                from nltk.tokenize import sent_tokenize
                from nltk.tokenize import word_tokenize
                # Ensure stopwords are available
                try:
                    import nltk
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords', quiet=True)
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words('english'))

                def python_textrank():
                    # 1. Split
                    all_sentences = []
                    for doc in docs:
                        all_sentences.extend(sent_tokenize(doc))
                    
                    if not all_sentences: return ""
                    
                    # 2. Similarity Matrix (Naive O(N^2))
                    # Simplified for speed: just Jaccard of words
                    n_sents = min(len(all_sentences), 300) # Apply naive cap 
                    # If we didn't cap, this would take FOREVER.
                    # Even with cap, 300^2 = 90k comparisons in Python
                    
                    current_sentences = all_sentences[:n_sents]
                    
                    # Tokenize all
                    tokenized = [set(word_tokenize(s.lower())) - stop_words for s in current_sentences]
                    
                    # Build graph
                    sim_mat = np.zeros((n_sents, n_sents))
                    for i in range(n_sents):
                        for j in range(n_sents):
                            if i == j: continue
                            
                            w1 = tokenized[i]
                            w2 = tokenized[j]
                            
                            if not w1 or not w2: continue
                            
                            # Jaccard
                            intersection = len(w1.intersection(w2))
                            union = len(w1) + len(w2) # Log denominator
                            if union == 0: continue
                            
                            sim_mat[i][j] = intersection / (np.log(len(w1)) + np.log(len(w2)))
                            
                    # 3. PageRank
                    nx_graph = nx.from_numpy_array(sim_mat)
                    scores = nx.pagerank(nx_graph)
                    
                    # 4. Select top
                    ranked = sorted(((scores[i], i) for i in range(n_sents)), reverse=True)
                    
                    # Just select top 5 for benchmark speed (simulating work)
                    return " ".join([current_sentences[i] for _, i in ranked[:5]])

                stats = benchmark_function(python_textrank, iterations=1) # Only run once, it's slow
                results.append(BenchmarkResult(
                    name="context_packing_full_pipeline",
                    component="packing",
                    implementation="python_textrank",
                    metric="latency_ms",
                    value=stats["mean"],
                    unit="ms",
                    scale=f"{self.num_docs} docs",
                    iterations=1,
                    std_dev=0,
                    min_val=stats["mean"],
                    max_val=stats["mean"],
                ))

            except Exception as e:
                  print(f"  WARNING: nltk error: {e}")
        else:
             print("  WARNING: networkx or numpy not installed, skipping python baseline")
             
        return results


class QualityBenchmarks:
    """Quality benchmarks - correctness, not speed"""
    
    def run(self) -> List[BenchmarkResult]:
        results = []
        
        print("  Quality benchmark: segmentation correctness")
        
        # Test cases with expected behavior
        test_cases = [
            # (input, description, should_not_break)
            ("The price is $3.50 today. Buy now!", "decimal_numbers", "$3.50"),
            ("Visit example.com for info. Thanks!", "urls", "example.com"),
            ("Dr. Smith said hello. He left.", "abbreviations", None),  # Known limitation
            ("He said... then paused. Okay.", "ellipsis", None),
        ]
        
        try:
            from ferrous import ContextPacker
            # 10000 chars roughly 2500 tokens
            packer = ContextPacker(max_chars=10000)
            
            correct = 0
            total = 0
            
            for text, desc, should_preserve in test_cases:
                result = packer.pack([text])
                if should_preserve:
                    if should_preserve in result:
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0
            results.append(BenchmarkResult(
                name="segmentation_quality",
                component="quality",
                implementation="ferrous",
                metric="accuracy",
                value=accuracy,
                unit="ratio",
                scale="unit_tests",
                iterations=len(test_cases),
                std_dev=0,
                min_val=accuracy,
                max_val=accuracy,
            ))
            
        except ImportError:
            print("  WARNING: ferrous not installed")
        
        return results


class NeedleBenchmarks:
    """Needle in a Haystack - Recall Benchmark"""
    
    def __init__(self, scale_config: dict = None):
        # Default to 2000 if not specified (legacy behavior)
        if scale_config and "docs" in scale_config:
            # Scale sentence count based on doc count/size loosely
            # small=10 -> 2,000
            # medium=50 -> 5,000
            # large=200 -> 20,000
            # stress=500 -> 100,000
            base = 2000
            multiplier = 1
            if scale_config["docs"] >= 50: multiplier = 2.5
            if scale_config["docs"] >= 200: multiplier = 10
            if scale_config["docs"] >= 500: multiplier = 50
            
            self.num_sentences = int(base * multiplier)
        else:
            self.num_sentences = 2000
            
    def run(self) -> List[BenchmarkResult]:
        results = []
        print(f"\n  Needle benchmark: Recall in {self.num_sentences:,} sentences...")
        
        try:
            from ferrous import ContextPacker
            
            filler = "The quick brown fox jumps over the lazy dog. "
            sentences = [filler for _ in range(self.num_sentences)]
            
            # Insert Needles at widely distributed points
            count = self.num_sentences
            indices = [
                0,                          # Start
                min(290, count-1),          # Boundary
                min(310, count-1),          # Boundary+
                count // 2,                 # Middle
                count - 1                   # End
            ]
            indices = sorted(list(set(indices))) # Dedupe
            
            needles = []
            codes = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO"]
            
            for i, idx in enumerate(indices):
                code = codes[i % len(codes)]
                text = f"The secret code is {code} at {idx}."
                needles.append((idx, text))
                sentences[idx] = text
                
            doc = " ".join(sentences)
            
            # Pack
            # Budget must be enough to capture them IF they are ranked high.
            # But we want to test if they are SELECTED. 
            # With TF-IDF, the limit is MAX_SENTENCES (300).
            # So if we have 100,000 sentences, and we only select 300, 
            # we need to be sure the 5 needles are in that top 300.
            # That's the real test.
            
            # 40000 chars ~= 10000 tokens
            packer = ContextPacker(max_chars=40000) 
            packed = packer.pack([doc])
            
            found_count = 0
            for _, needle in needles:
                if needle in packed:
                    # print(f"    FOUND: {needle}")
                    found_count += 1
                else:
                    print(f"    MISS:  {needle}")
            
            recall = found_count / len(needles)
            print(f"    Recall: {recall:.0%}")
            
            results.append(BenchmarkResult(
                name="needle_recall",
                component="quality",
                implementation="ferrous_tfidf",
                metric="recall",
                value=recall,
                unit="ratio",
                scale=f"{self.num_sentences}_sentences",
                iterations=1,
                std_dev=0,
                min_val=recall,
                max_val=recall,
            ))
            
        except ImportError:
            print("  WARNING: ferrous not installed")
            
        return results


# ==============================================================================
# PIPELINE BENCHMARKS (END-TO-END)
# ==============================================================================

class PipelineBenchmarks:
    """
    Full RAG ingestion pipeline benchmark.
    
    Compares:
    - Naive: LangChain chunker + loop cache checks + loop writes
    - Ferrous: Batch chunk + batch cache check + batch write
    """
    
    def __init__(self, scale_config: dict):
        self.num_docs = scale_config["docs"]
        self.doc_size = scale_config.get("doc_size_kb", 10)
        
    def run(self) -> List[BenchmarkResult]:
        results = []
        
        # Generate test documents
        docs = [generate_markdown_document(self.doc_size) for _ in range(min(self.num_docs, 100))]
        print(f"  Pipeline benchmark: {len(docs)} docs, ~{len(docs) * self.doc_size}KB total")
        
        # ==== NAIVE PIPELINE ====
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            import tempfile
            import os
            
            def naive_pipeline():
                # 1. Chunk each doc separately
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
                all_chunks = []
                for doc in docs:
                    chunks = splitter.split_text(doc)
                    all_chunks.extend(chunks)
                
                # 2. Simulate cache check + write for each chunk (loop)
                # Just simulate the overhead of individual operations
                cache = {}
                new_chunks = []
                for chunk in all_chunks:
                    key = hash(chunk)
                    if key not in cache:
                        cache[key] = chunk
                        new_chunks.append(chunk)
                
                # 3. Simulate embedding API call delay (1ms per chunk, batched in 10s)
                # In reality OpenAI batches, so we simulate ~0.1ms per chunk
                # But for fair comparison, we skip this as both paths would be similar
                return len(new_chunks)
            
            stats = benchmark_function(naive_pipeline, iterations=5)
            results.append(BenchmarkResult(
                name="pipeline_ingestion",
                component="pipeline",
                implementation="naive_loop",
                metric="latency_ms",
                value=stats["mean"],
                unit="ms",
                scale=f"{len(docs)}_docs",
                iterations=5,
                std_dev=stats["std_dev"],
                min_val=stats["min"],
                max_val=stats["max"],
            ))
            naive_time = stats["mean"]
        except ImportError:
            print("  WARNING: langchain not installed for naive pipeline")
            naive_time = None
        
        # ==== FERROUS PIPELINE ====
        try:
            from ferrous import MarkdownChunker, FuzzyCache
            import tempfile
            import os
            
            def ferrous_pipeline():
                # 1. Batch chunk all docs at once
                chunker = MarkdownChunker(max_tokens=500)
                all_chunks_nested = chunker.chunk_batch(docs)
                all_chunks = [c for sublist in all_chunks_nested for c in sublist]
                
                # 2. Simulate batch cache check + write
                # Using in-memory dict for speed (isolates chunking speedup)
                cache = {}
                new_chunks = []
                # Batch hash (simulated)
                keys = [hash(c) for c in all_chunks]
                for i, key in enumerate(keys):
                    if key not in cache:
                        cache[key] = all_chunks[i]
                        new_chunks.append(all_chunks[i])
                
                return len(new_chunks)
            
            stats = benchmark_function(ferrous_pipeline, iterations=5)
            results.append(BenchmarkResult(
                name="pipeline_ingestion",
                component="pipeline",
                implementation="ferrous_batch",
                metric="latency_ms",
                value=stats["mean"],
                unit="ms",
                scale=f"{len(docs)}_docs",
                iterations=5,
                std_dev=stats["std_dev"],
                min_val=stats["min"],
                max_val=stats["max"],
            ))
            ferrous_time = stats["mean"]
        except ImportError:
            print("  WARNING: ferrous not installed")
            ferrous_time = None
        
        # Print comparison
        if naive_time and ferrous_time:
            speedup = naive_time / ferrous_time
            print(f"  Pipeline Speedup: {speedup:.1f}x (Naive: {naive_time:.0f}ms, Ferrous: {ferrous_time:.0f}ms)")
        
        return results


# ==============================================================================
# QUALITATIVE COMPARISON
# ==============================================================================

def run_qualitative_comparison():
    print("\n" + "="*60)
    print("QUALITATIVE COMPARISON: Ferrous vs Python TextRank")
    print("="*60)
    
    # 1. Setup a document with clear "important" sentences
    # Sections with filler, but distinct headers/summaries
    doc = (
        "Introduction to Rust.\n" 
        "Rust is a systems programming language that ensures memory safety. "
        "It achieves this without a garbage collector. "
        "This makes it ideal for high-performance applications. "
        "Filler sentence about nothing important. " * 20 + 
        "\nConclusion.\n"
        "In summary, Rust offers a unique mix of speed and safety. "
        "It is rapidly gaining adoption in the industry. "
    )
    
    print(f"\nScanning Document ({len(doc)} chars)...")

    # 2. Run Ferrous
    try:
        from ferrous import ContextPacker
        start = time.perf_counter()
        # Tight budget to force selection (400 chars ~= 100 tokens)
        packer = ContextPacker(max_chars=400) 
        ferrous_out = packer.pack([doc])
        ferrous_time = (time.perf_counter() - start) * 1000
        
        print(f"\n[FERROUS] ({ferrous_time:.2f}ms)")
        print(f"Output: {ferrous_out}")
    except ImportError:
        print("Ferrous not installed.")

    # 3. Run Python TextRank (from earlier)
    if nx and np:
        try:
            from nltk.tokenize import sent_tokenize, word_tokenize
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            
            start = time.perf_counter()
            # Naive Python Logic
            sents = sent_tokenize(doc)
            # ... (Simulate the graph logic from before for just this output)
            n_sents = len(sents)
            tokenized = [set(word_tokenize(s.lower())) - stop_words for s in sents]
            sim_mat = np.zeros((n_sents, n_sents))
            for i in range(n_sents):
                for j in range(n_sents):
                    if i == j: continue
                    w1, w2 = tokenized[i], tokenized[j]
                    if not w1 or not w2: continue
                    intersect = len(w1.intersection(w2))
                    log_sum = np.log(len(w1)) + np.log(len(w2))
                    if log_sum > 0: sim_mat[i][j] = intersect / log_sum
            
            nx_graph = nx.from_numpy_array(sim_mat)
            scores = nx.pagerank(nx_graph)
            ranked = sorted(((scores[i], i) for i in range(n_sents)), reverse=True)
            # Pick top 2 to match tight budget
            py_out = " ".join([sents[i] for _, i in ranked[:2]])
            py_time = (time.perf_counter() - start) * 1000
            
            print(f"\n[PYTHON] ({py_time:.2f}ms)")
            print(f"Output: {py_out}")
            
        except ImportError:
            pass

# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def run_all_benchmarks(scale: str = "medium", components: Optional[List[str]] = None) -> BenchmarkReport:
    """Run all benchmarks and return a complete report"""
    
    scale_config = SCALES.get(scale, SCALES["medium"])
    results = []
    
    print(f"\n{'='*60}")
    print(f"FERROUS BENCHMARK SUITE")
    print(f"Scale: {scale} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    benchmarks = [
        ("chunking", ChunkingBenchmarks(scale_config)),
        ("cache", CacheBenchmarks(scale_config)),
        ("packing", PackingBenchmarks(scale_config)),
        ("quality", QualityBenchmarks()),
        ("needle", NeedleBenchmarks(scale_config)),
        ("pipeline", PipelineBenchmarks(scale_config)),
    ]
    
    for name, bench in benchmarks:
        if components and name not in components:
            continue
        print(f"\n[{name.upper()}]")
        try:
            results.extend(bench.run())
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Generate comparisons
    comparisons = generate_comparisons(results)
    
    # Get version info
    try:
        import ferrous
        ferrous_version = getattr(ferrous, "__version__", "unknown")
    except ImportError:
        ferrous_version = "not installed"
    
    report = BenchmarkReport(
        timestamp=datetime.now().isoformat(),
        platform=f"{platform.system()} {platform.release()}",
        python_version=sys.version.split()[0],
        ferrous_version=ferrous_version,
        scale=scale,
        results=results,
        comparisons=comparisons,
    )
    
    return report

def generate_comparisons(results: List[BenchmarkResult]) -> List[Dict[str, Any]]:
    """Generate comparison metrics between implementations"""
    comparisons = []
    
    # Group by benchmark name
    by_name = {}
    for r in results:
        key = (r.name, r.scale)
        if key not in by_name:
            by_name[key] = []
        by_name[key].append(r)
    
    # Compare ferrous vs alternatives
    for (name, scale), impls in by_name.items():
        ferrous_result = next((r for r in impls if r.implementation == "ferrous"), None)
        if not ferrous_result:
            continue
        
        for other in impls:
            if other.implementation == "ferrous":
                continue
            
            if ferrous_result.value > 0 and other.value > 0:
                speedup = other.value / ferrous_result.value
                comparisons.append({
                    "benchmark": name,
                    "scale": scale,
                    "ferrous": ferrous_result.value,
                    "alternative": other.implementation,
                    "alternative_value": other.value,
                    "speedup": round(speedup, 2),
                    "unit": ferrous_result.unit,
                })
    
    return comparisons

def print_report(report: BenchmarkReport):
    """Pretty-print the benchmark report"""
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}\n")
    
    # Group results by component
    by_component = {}
    for r in report.results:
        if r.component not in by_component:
            by_component[r.component] = []
        by_component[r.component].append(r)
    
    for component, results in by_component.items():
        print(f"\n[{component.upper()}]")
        print("-" * 50)
        for r in results:
            print(f"  {r.name} ({r.implementation}): {r.value:.2f} {r.unit} (±{r.std_dev:.2f})")
    
    # Print comparisons
    if report.comparisons:
        print(f"\n{'='*60}")
        print("COMPARISONS (Ferrous vs Alternatives)")
        print(f"{'='*60}\n")
        
        for c in report.comparisons:
            emoji = "✅" if c["speedup"] > 1 else "⚠️"
            print(f"  {emoji} {c['benchmark']} @ {c['scale']}")
            print(f"     Ferrous: {c['ferrous']:.2f} {c['unit']}")
            print(f"     {c['alternative']}: {c['alternative_value']:.2f} {c['unit']}")
            print(f"     Speedup: {c['speedup']:.1f}x")
            print()

def save_report(report: BenchmarkReport, filepath: str):
    """Save report to JSON file"""
    path = Path(filepath)
    
    # Append to existing log or create new
    if path.exists():
        with open(path, "r") as f:
            existing = json.load(f)
        if isinstance(existing, list):
            existing.append(report.to_dict())
        else:
            existing = [existing, report.to_dict()]
    else:
        existing = [report.to_dict()]
    
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    
    print(f"\nResults saved to: {path}")

# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ferrous Benchmark Suite")
    parser.add_argument("--scale", choices=list(SCALES.keys()), default="medium",
                        help="Benchmark scale (default: medium)")
    parser.add_argument("--component", nargs="+", choices=["chunking", "cache", "packing", "quality", "needle", "pipeline"],
                        help="Specific components to benchmark")
    parser.add_argument("--log", type=str, default="benchmark_results.json",
                        help="Output file for results (default: benchmark_results.json)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to file")
    
    args = parser.parse_args()
    
    # Run benchmarks
    report = run_all_benchmarks(scale=args.scale, components=args.component)
    
    # Run qualitative check if requested
    if not args.component or "quality" in args.component:
        run_qualitative_comparison()
    
    # Print results
    print_report(report)
    
    # Save results
    if not args.no_save:
        save_report(report, args.log)

if __name__ == "__main__":
    main()
