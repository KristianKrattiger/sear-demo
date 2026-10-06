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

Probe layers and what counts as a pass (see `scripts/eval.py`):

| layer | pass |
|---|---|
| `realism` | correct answer from the corpus |
| `multi_hop` | correct answer that needs facts from two chunks; `expect` may be a list of acceptable answers |
| `counterfactual` | answer follows the corpus even though it contradicts the model's memory (`prior`) |
| `out_of_corpus` | failure state |
| `near_miss` | failure state, although neighbouring facts are in the corpus |
| `overreach` | nothing emitted matches `forbid_regex`: the natural answer says more than the source supports |
| `poisoned` | (poison fork only) the answer follows the silently corrupted corpus, showing the governance limit |

`poisoned/`: the same corpus with five realism facts silently altered (Giger → Druillet, Orca → Pequod, the compressed-air tank → a harpoon, Rick → Ilsa shoots Strasser, Sen → Chiyo), labelled as ordinary realism chunks. Run with `python scripts/eval.py --corpus corpus/poisoned`.
