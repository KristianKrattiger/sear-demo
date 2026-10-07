# sear-demo design notes

## Audience and goal

A live demo for Prof. Liu. It should be short, falsifiable, and honest about its limits. The core claim: a frozen model wrapped in SEAR emits only claims traceable to the corpus, and says "the corpus does not support this" when it can't ground an answer.

## Demo script (about 8 minutes)

1. **In-corpus question.** A cited answer. Click a citation and the chunk opens.
2. **Counterfactual probe.** The corpus has been altered on purpose (for example a changed release year or director). The answer follows the corpus, not the model's memory. This is the surprise beat.
3. **Out-of-corpus question.** The failure state fires.
4. **Narrator on.** Re-run question 1 with the switch on. The readable summary comes first and the grounded report stays underneath. Point at the per-sentence verification badges.
5. **Close (optional): poison fork.** Same question on `corpus/poisoned/`. You get a fully cited, verified, wrong answer. That is why GIN needs governance and not just engineering.
6. **Numbers.** The eval table: fabrication rate, counterfactual adherence, failure-state precision/recall, and the narrator's verification rate.

## Pipeline and narrator rules

These now live in [ARCHITECTURE.md](ARCHITECTURE.md), which follows the current code: retrieval, citation-group segmentation, verification by NLI plus the names-and-numbers check, citation repair, the gate, and the narrator's handles, labels and source chains. In short, the narrator sees only the verified report as one handle per sentence, cites those handles, and every summary sentence is re-checked against the handles it cites and badged when a check fails.

## Default choices (change freely)

| Choice | Default | Why |
|---|---|---|
| Corpus domain | Film | Recommended by ENG 01; you know it cold; supports a later divergent mode |
| Corpus size | 200–500 chunks to start | Enough to test the mechanics |
| Chunk size | 150–300 tokens, natural boundaries, head sentence | Matches the anchor design in ENG 01 §2b |
| Embeddings | `BAAI/bge-m3` | ~0.6B, MIT, strong retrieval, leaves VRAM for the generator |
| Vector store | FAISS (flat inner product) | Simple and local |
| Generator | `microsoft/phi-4` (14B, 4-bit), HF `transformers` | ~9 GB on the 4070, MIT, high faithfulness; fallback Qwen3 8B if citation format is weak |
| Comparison model | `meta-llama/Llama-3.2-3B-Instruct` | Small model, more corpus in view (Stage 1c comparison) |
| Verifier | `cross-encoder/nli-deberta-v3-base` | A dedicated NLI model rather than an LLM judge |
| Narrator | Pluggable: local model or an API model | It doesn't need attention logging |
| UI | Streamlit | The fastest way to a clean demo |
| τ (entailment threshold) | 0.5 | TBM |
| φ (min verified fraction) | 0.5 | TBM |

## Open questions

- Local or API for the narrator during the demo? An API model is more fluent, and a local one is fully offline.
- Which counterfactuals make the best "surprise" beat? They should be facts the model is very confident about.
- Should the poison fork be in the live demo or only in the slides?
