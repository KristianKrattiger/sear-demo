# sear-demo

A small, honest demo of **SEAR** (Sparse Epistemically Anchored Reasoning), the constrained-inference layer of GIN.

It shows three things on a hand-built film corpus:

1. **Grounded answers.** Every claim cites a corpus chunk, and a verifier checks that the chunk actually supports it.
2. **Corpus over memory.** On counterfactual probes, the answer follows the corpus even when the model "knows" otherwise.
3. **An explicit failure state.** When the corpus can't support an answer, SEAR says so instead of guessing.

An optional **narrator** (off by default) turns the dense grounded report into a readable summary. It sees only the verified report, cites only the report's chunk IDs, and its sentences are re-verified. The original grounded report is always shown underneath.

This is ENG 01 Stages 0–1 plus the Stage 4 narrator pulled forward. Nothing here is a measured spec yet (see the GIN ENG 00 promotion rule).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
python scripts/build_index.py          # embeds corpus/chunks.jsonl into a FAISS index
streamlit run app/app.py               # the demo UI
python scripts/eval.py                 # runs the probe set and prints metrics
```

Settings live in `config/default.yaml`. The narrator is `narrator.enabled: false` by default and can also be flipped in the UI.

## Layout

```
config/            default.yaml (models, top-k, thresholds, narrator switch)
corpus/            chunks.jsonl (stable chunk IDs) and probes.jsonl (three probe layers)
src/sear_demo/     pipeline: retrieve → generate → segment → verify → gate → (narrator)
app/               Streamlit UI
scripts/           index build and eval
docs/DESIGN.md     demo script, architecture, decisions, open questions
```
