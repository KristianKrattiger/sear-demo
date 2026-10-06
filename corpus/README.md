# Corpus

`chunks.jsonl`: one chunk per line. IDs are assigned once at build time and never change.

```json
{"id": "f0001-c03", "film": "...", "head": "Most representative sentence.", "text": "Full chunk, 150-300 tokens, natural boundary.", "layer": "realism"}
```

`layer` is `realism` or `counterfactual`. Counterfactual chunks alter a verifiable detail on purpose; record the real value in `probes.jsonl`, not in the chunk.

`probes.jsonl`: one evaluation probe per line.

```json
{"id": "p001", "layer": "realism|counterfactual|out_of_corpus", "question": "...", "expect": "corpus answer, or null for out_of_corpus", "prior": "what the model would say from memory (counterfactual only)"}
```

`poisoned/` (later): a fork of chunks.jsonl for the poison demo.
