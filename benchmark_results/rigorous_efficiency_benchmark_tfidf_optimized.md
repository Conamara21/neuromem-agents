# Rigorous Efficiency Benchmark

## Methodology
- Real package implementations were benchmarked, not mock classes.
- Each trial ran in an isolated subprocess with `PYTHONHASHSEED=0` and single-thread BLAS settings.
- Exact-match retrieval is used as a deterministic sanity check.
- Topic queries test whether shared lexical/topic signals retrieve documents from the correct cluster.
- A separate primed associative workload is kept as an exploratory signal for association dynamics.

## Configuration
- Trials per condition: 3
- Sizes: 128, 512, 1024
- Embedding backend: tfidf
- Query count per trial: 64
- Warmup count per trial: 16
- Association degree: 3

## Summary
### Corpus Size 128
- `traditional_rag`
  build 0.126997s +/- 0.004881s, exact p95 3.813 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 0.850, heap 739853 B, db 0 B
- `neuromem_in_memory`
  build 0.146778s +/- 0.005017s, exact p95 3.714 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 1.000, heap 1222732 B, db 0 B
- `neuromem_persistent`
  build 0.204469s +/- 0.026370s, exact p95 3.823 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 1.000, heap 1255909 B, db 946176 B

### Corpus Size 512
- `traditional_rag`
  build 0.503083s +/- 0.014651s, exact p95 14.322 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 0.345, heap 8758095 B, db 0 B
- `neuromem_in_memory`
  build 0.585035s +/- 0.015404s, exact p95 3.852 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 0.854, heap 10493598 B, db 0 B
- `neuromem_persistent`
  build 0.825224s +/- 0.008465s, exact p95 3.905 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 0.854, heap 10550764 B, db 9351168 B

### Corpus Size 1024
- `traditional_rag`
  build 0.988829s +/- 0.038051s, exact p95 26.152 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 0.059, heap 17473953 B, db 0 B
- `neuromem_in_memory`
  build 1.186357s +/- 0.048207s, exact p95 5.811 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 0.132, heap 21104272 B, db 0 B
- `neuromem_persistent`
  build 1.714646s +/- 0.073083s, exact p95 7.964 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 0.132, heap 21175660 B, db 18602667 B

## Caveats
- `tfidf` gives interpretable lexical retrieval, but it is still not a semantic encoder; paraphrase/generalization claims would require a stronger shared model.
- The primed neighbor-recall metric is exploratory and is only useful for checking whether the association machinery changes ranking behavior at all.
- TraditionalRAGSystem is currently an in-memory baseline without persistence; persistent NeuroMem measurements therefore include a feature cost that the baseline does not bear.
