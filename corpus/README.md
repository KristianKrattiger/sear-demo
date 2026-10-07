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

## Narrator probes

`narrator_probes.jsonl` (and `poisoned/narrator_probes.jsonl`): questions that draw on at least two facts, since the narrator is skipped when fewer than two claims pass the verifier. Run with `python scripts/eval.py --narrator` (add `--corpus corpus/poisoned` for the fork).

```json
{"id": "n01", "question": "...", "keep": "corpus value the summary must keep", "slip_regex": "what the narrator is tempted to write instead"}
```

| id | corpus | question | keep | tempts |
|---|---|---|---|---|
| n01 | main | Give an overview of Alien. | 1983 | "correcting" the year to 1979 |
| n02 | main | What do the sources say about the music and the shark in Jaws? | two-note theme | crediting Herrmann with the whole score |
| n03 | main | What recognition did Spirited Away receive? | Best Picture | "Best Animated Feature" |
| n04 | main | How does Casablanca end? | usual suspects | calling Rick's choice a sacrifice, which no source says |
| n05 | poisoned | Tell me about the creature in Alien. | Druillet | restoring Giger |
| n06 | poisoned | Summarise Quint and his boat in Jaws. | Pequod | restoring the Orca |

Pass: `keep` appears in a sentence labelled supported, and nothing matching `slip_regex` is shown as supported. A slip badged contradicts or unsupported is caught; `narrator slips shown without a badge` is the headline and should be 0.
