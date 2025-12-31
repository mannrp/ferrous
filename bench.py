import time
import os
import json
from ferrous import MarkdownChunker, FuzzyCache, ContextPacker

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        HAS_LANGCHAIN = True
    except ImportError:
        HAS_LANGCHAIN = False

def bench_fuzzy_cache():
    print("\n--- Benchmark 1: Fuzzy Cache (SimHash) ---")
    cache_path = "bench_cache.db"
    if os.path.exists(cache_path): 
        try: os.remove(cache_path)
        except: pass
    
    # Use higher threshold and smaller shingle for shorter text
    cache = FuzzyCache(cache_path, threshold=10, shingle_size=2)
    query = "What is the historical revenue of the company in Q3 2023?"
    data = json.dumps({"result": "4.2 Billion USD", "source": "Annual Report"})

    # Set
    cache.put(query, data)
    
    # Near-Match (SimHash is invariant to small lexical changes)
    typo_query = "What is the historical revenu of the company in Q3 2023?" # spelling fix
    
    start = time.perf_counter_ns()
    hit = cache.get(typo_query)
    end = time.perf_counter_ns()
    
    latency_us = (end - start) / 1000
    print(f"Near-duplicate Lookup Latency: {latency_us:.2f} µs")
    print(f"Result: {'HIT' if hit else 'MISS'}")

def bench_markdown_chunker():
    print("\n--- Benchmark 2: Markdown Chunking ---")
    # 500kb markdown file
    content = "# Main Header\n\n" + "This paragraph exists to fill space and test speed. " * 3000 + "\n\n## Subheader\n" + "More filler text here. " * 2000
    
    print(f"Payload Size: {len(content) / 1024:.2f} KB")

    # Ferrous
    start = time.perf_counter()
    chunker = MarkdownChunker(max_tokens=1000)
    ferrous_chunks = chunker.chunk(content)
    ferrous_time = (time.perf_counter() - start) * 1000
    print(f"Ferrous (Rust): {ferrous_time:.2f} ms")

    # LangChain
    if HAS_LANGCHAIN:
        start = time.perf_counter()
        lc_chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        lc_chunks = lc_chunker.split_text(content)
        lc_time = (time.perf_counter() - start) * 1000
        print(f"LangChain (Py): {lc_time:.2f} ms")
        print(f"Speedup: {lc_time / ferrous_time:.1f}x")
    else:
        print("LangChain not available for comparison.")

def bench_context_packer():
    print("\n--- Benchmark 3: Context Packing (TextRank) ---")
    # Simulate retrieving 50 chunks that need packing
    docs = [f"This is document segment {i}. It contains some duplicate information about the sun being a star. The sun is a yellow dwarf star at the center of the solar system. " * 5 for i in range(50)]
    
    start = time.perf_counter()
    packer = ContextPacker(max_tokens=2000)
    packed = packer.pack(docs)
    end = time.perf_counter()
    
    print(f"Packer Latency (50 docs -> 2000 chars): {(end-start)*1000:.2f} ms")
    print(f"Reduction: {len(json.dumps(docs)) / len(packed):.1f}x compression by importance")

if __name__ == "__main__":
    try:
        bench_fuzzy_cache()
        bench_markdown_chunker()
        bench_context_packer()
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        print("Note: Ensure you have built the library with 'maturin develop --release'")
