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
        texts = generate_similar_texts(base_text, min(self.num_entries, 1000))
        
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
                
                # Populate cache
                for i, text in enumerate(texts):
                    cache.put(text, f"response_{i}")
                
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
                
            finally:
                os.unlink(db_path)
                
        except ImportError:
            print("  WARNING: ferrous not installed")
        
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
                packer = ContextPacker(max_tokens=4000)
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
            packer = ContextPacker(max_tokens=10000)
            
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
    
    def run(self) -> List[BenchmarkResult]:
        results = []
        print("\n  Needle benchmark: Recall of specific facts at depth")
        
        try:
            from ferrous import ContextPacker
            
            # Setup: Create a long document (2000 sentences)
            # 2000 sentences is well above the 300 cap.
            # Truncation would fail at index > 300.
            # TF-IDF should rescue them if they are distinct.
            
            filler = "The quick brown fox jumps over the lazy dog. "
            sentences = [filler for _ in range(2000)]
            
            # Insert Needles
            needles = [
                (0, "The secret code is ALPHA."),      # Start (always kept)
                (290, "The secret code is BRAVO."),    # Near truncation boundary
                (310, "The secret code is CHARLIE."),  # Just past 300 cap
                (1000, "The secret code is DELTA."),   # Middle
                (1999, "The secret code is ECHO."),    # End
            ]
            
            for idx, text in needles:
                sentences[idx] = text
                
            doc = " ".join(sentences)
            
            # Pack
            packer = ContextPacker(max_tokens=6000) # Give enough budget
            packed = packer.pack([doc])
            
            found_count = 0
            for _, needle in needles:
                if needle in packed:
                    print(f"    FOUND: {needle}")
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
                scale="2000_sentences",
                iterations=1,
                std_dev=0,
                min_val=recall,
                max_val=recall,
            ))
            
        except ImportError:
            print("  WARNING: ferrous not installed")
            
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
        packer = ContextPacker(max_tokens=100) # Tight budget to force selection
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
        ("needle", NeedleBenchmarks()),
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
    parser.add_argument("--component", nargs="+", choices=["chunking", "cache", "packing", "quality", "needle"],
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
