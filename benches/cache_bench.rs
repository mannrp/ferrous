use criterion::{black_box, criterion_group, criterion_main, Criterion};
use _ferrous::cache::SimHash;
use _ferrous::chunking::MarkdownChunker;
use _ferrous::packing::ContextPacker;

fn bench_simhash(c: &mut Criterion) {
    let hasher = SimHash::new(3);
    let text = "The quick brown fox jumps over the lazy dog".repeat(10);
    c.bench_function("simhash_fingerprint_short", |b| b.iter(|| hasher.fingerprint(black_box(&text))));
}

fn bench_markdown_chunker(c: &mut Criterion) {
    let chunker = MarkdownChunker::new(500);
    let md = "# Header\n".to_owned() + &"This is a paragraph of some length. ".repeat(100);
    c.bench_function("markdown_chunk_long", |b| b.iter(|| chunker.chunk(black_box(&md))));
}

fn bench_context_packer(c: &mut Criterion) {
    let packer = ContextPacker::new(1000);
    let docs = vec![
        "Document one content. ".repeat(50),
        "Document two content. ".repeat(50),
        "Document three content. ".repeat(50),
    ];
    c.bench_function("context_packer_3_docs", |b| b.iter(|| packer.pack(black_box(&docs))));
}

criterion_group!(benches, bench_simhash, bench_markdown_chunker, bench_context_packer);
criterion_main!(benches);
