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
  build 0.123922s +/- 0.004923s, exact p95 3.910 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 0.850, heap 739745 B, db 0 B
- `neuromem_in_memory`
  build 0.129123s +/- 0.004653s, exact p95 5.942 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 1.000, heap 777416 B, db 0 B
- `neuromem_persistent`
  build 4.040614s +/- 0.022015s, exact p95 33.027 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 1.000, heap 793746 B, db 913408 B

### Corpus Size 512
- `traditional_rag`
  build 0.550730s +/- 0.030343s, exact p95 17.067 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 0.345, heap 8757588 B, db 0 B
- `neuromem_in_memory`
  build 0.551507s +/- 0.013817s, exact p95 21.569 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 0.854, heap 8892216 B, db 0 B
- `neuromem_persistent`
  build 16.103802s +/- 0.150244s, exact p95 70.824 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 0.854, heap 8935400 B, db 9318400 B

### Corpus Size 1024
- `traditional_rag`
  build 1.010847s +/- 0.022214s, exact p95 30.023 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 0.059, heap 17467624 B, db 0 B
- `neuromem_in_memory`
  build 1.076749s +/- 0.060698s, exact p95 41.386 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 0.132, heap 17753257 B, db 0 B
- `neuromem_persistent`
  build 32.073827s +/- 0.146739s, exact p95 85.426 ms, top1 1.000, topic-hit@5 1.000, neighbor recall 0.132, heap 17808209 B, db 18568533 B

## Caveats
- `tfidf` gives interpretable lexical retrieval, but it is still not a semantic encoder; paraphrase/generalization claims would require a stronger shared model.
- The primed neighbor-recall metric is exploratory and is only useful for checking whether the association machinery changes ranking behavior at all.
- TraditionalRAGSystem is currently an in-memory baseline without persistence; persistent NeuroMem measurements therefore include a feature cost that the baseline does not bear.
