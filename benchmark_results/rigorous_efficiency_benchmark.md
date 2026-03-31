# Rigorous Efficiency Benchmark

## Methodology
- Real package implementations were benchmarked, not mock classes.
- Each trial ran in an isolated subprocess with `PYTHONHASHSEED=0` and single-thread BLAS settings.
- Exact-match retrieval was used as the primary functional sanity check because current embeddings are hash-based placeholders.
- A separate primed associative workload was kept as an exploratory signal only; with the current hash-based embeddings it should not be treated as a final retrieval-quality claim.

## Configuration
- Trials per condition: 3
- Sizes: 128, 512, 1024
- Query count per trial: 64
- Warmup count per trial: 16
- Association degree: 3

## Summary
### Corpus Size 128
- `traditional_rag`
  build 0.011196s +/- 0.000018s, exact p95 2.654 ms, top1 1.000, neighbor recall 0.023, heap 145812 B, db 0 B
- `neuromem_in_memory`
  build 0.015194s +/- 0.000226s, exact p95 2.980 ms, top1 0.792, neighbor recall 0.056, heap 164936 B, db 0 B
- `neuromem_persistent`
  build 3.748117s +/- 0.039092s, exact p95 29.335 ms, top1 0.792, neighbor recall 0.056, heap 179702 B, db 335872 B

### Corpus Size 512
- `traditional_rag`
  build 0.046978s +/- 0.003750s, exact p95 12.433 ms, top1 1.000, neighbor recall 0.012, heap 566804 B, db 0 B
- `neuromem_in_memory`
  build 0.057150s +/- 0.001860s, exact p95 3.076 ms, top1 0.198, neighbor recall 0.013, heap 400504 B, db 0 B
- `neuromem_persistent`
  build 14.846289s +/- 0.110099s, exact p95 30.016 ms, top1 0.198, neighbor recall 0.013, heap 414401 B, db 1149611 B

### Corpus Size 1024
- `traditional_rag`
  build 0.088982s +/- 0.000445s, exact p95 21.430 ms, top1 1.000, neighbor recall 0.009, heap 1129061 B, db 0 B
- `neuromem_in_memory`
  build 0.112323s +/- 0.001500s, exact p95 3.013 ms, top1 0.120, neighbor recall 0.013, heap 710280 B, db 0 B
- `neuromem_persistent`
  build 30.357481s +/- 0.204322s, exact p95 27.986 ms, top1 0.120, neighbor recall 0.013, heap 725927 B, db 2252800 B

## Caveats
- The current retrieval embedding is a placeholder derived from Python hashes, so semantic retrieval quality benchmarks would not be scientifically meaningful yet.
- The primed neighbor-recall metric is exploratory and is only useful for checking whether the association machinery changes ranking behavior at all.
- TraditionalRAGSystem is currently an in-memory baseline without persistence; persistent NeuroMem measurements therefore include a feature cost that the baseline does not bear.
