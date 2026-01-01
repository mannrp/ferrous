import time
from ferrous import FuzzyCache, ContextPacker, MarkdownChunker

def simulate_expensive_call(text):
    time.sleep(0.1)  # Simulate 100ms API latency
    return f"Embedding for: {text[:20]}..."

def main():
    print("=== Ferrous Demo ===\n")

    # 1. Fuzzy Cache Demo
    print("--- Fuzzy Cache ---")
    cache = FuzzyCache("demo_cache.db", threshold=15, shingle_size=3)
    
    query = "What is the capital of France?"
    similar_query = "What's the capital of France?" # Slight variation

    # First call - Miss
    start = time.time()
    if not cache.get(query):
        result = simulate_expensive_call(query)
        cache.put(query, result)
        print(f"1. Query: '{query}' -> MISS (Computed in {time.time()-start:.4f}s)")
    
    # Second call - Hit (Exact or Fuzzy)
    start = time.time()
    cached = cache.get(similar_query)
    if cached:
        print(f"2. Query: '{similar_query}' -> HIT (Retrieved in {time.time()-start:.4f}s)")
        print(f"   Matches cached data: {cached}")

    # 2. Context Packing Demo
    print("\n--- Context Packer ---")
    packer = ContextPacker(max_tokens=100) # Small budget
    
    docs = [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "Paris is the capital and most populous city of France.",
        "The tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building.",
        "It was named after the engineer Gustave Eiffel, whose company designed and built the tower."
    ]
    
    # Pack without redundancy
    start = time.time()
    packed = packer.pack(docs)
    print(f"Packed {len(docs)} docs into {len(packed.split())} words in {time.time()-start:.4f}s:")
    print(f"---\n{packed}\n---")

if __name__ == "__main__":
    main()
