# sear-demo design notes

## Audience and goal

A live demo for Prof. Liu. It should be short, falsifiable, and honest about its limits. The core claim: a frozen model wrapped in SEAR emits only claims traceable to the corpus, and says "the corpus does not support this" when it can't ground an answer.

## Demo script (about 8 minutes)

1. **In-corpus question.** "Who designed the creature in Alien?" A cited answer. Click the chunk ID under the claim and the chunk opens.
2. **Counterfactual probe.** Three chunks were altered on purpose: Alien's release year (1983), the composer of Jaws's two-note theme (Bernard Herrmann) and Spirited Away's Oscar (Best Picture). Ask "What year was Alien released?" The answer follows the corpus, not the model's memory (1979). This is the surprise beat.
3. **Out-of-corpus question.** "What was the budget of Alien?" The failure state fires.
4. **Narrator on.** Turn the switch on and ask "Give an overview of Spirited Away, including the awards it won." (Question 1's answer is a single sentence, and the narrator skips a report with fewer than two verified sentences.) The readable summary comes first and the grounded report stays underneath. Point at the per-sentence badges and open a Sources expander.
5. **Close (optional): poisoned fork.** Pick "poisoned" in the corpus selector (only the chunks and the index change, so it takes seconds) and ask question 1 again: a fully cited, verified, wrong answer (Druillet). Or turn on "Compare corpora" and ask once to see both answers side by side with their citations and entailment scores. That is why GIN needs governance and not just engineering. On the "conflict" fork, "What is the name of Quint's boat in Jaws?" gets a verified "Orca" although another chunk says Pequod: the answer picks one value silently.
6. **Numbers.** The Results section of the README: fabrication rate, counterfactual adherence, failure-state precision and recall, the narrator's verification rate, and the controls by type with what slipped through, each with the commit its run came from.

## Pipeline and narrator rules

These now live in [ARCHITECTURE.md](ARCHITECTURE.md), which follows the current code: retrieval, citation-group segmentation, verification by NLI plus the names-and-numbers check, citation repair, the gate, and the narrator's handles, labels and source chains. In short, the narrator sees only the verified report as one handle per sentence, cites those handles, and every summary sentence is re-checked against the handles it cites and badged when a check fails.

## Default choices (change freely)

| Choice | Default | Why |
|---|---|---|
| Corpus domain | Film | Recommended by ENG 01; you know it cold; supports a later divergent mode |
| Corpus size | 13 hand-written chunks on four films, plus a poisoned fork and a conflict fork | Enough to show the mechanics. ENG 01's 200–500 chunks is the next step; at 13, `top_k` 12 means retrieval filters almost nothing |
| Chunk size | 150–300 tokens, natural boundaries, head sentence | Matches the anchor design in ENG 01 §2b |
| Embeddings | `BAAI/bge-m3` | ~0.6B, MIT, strong retrieval, leaves VRAM for the generator |
| Vector store | FAISS (flat inner product) | Simple and local |
| Generator | `microsoft/phi-4` (14B, 4-bit), HF `transformers` | ~9 GB on the 4070, MIT, high faithfulness; fallback Qwen3 8B if citation format is weak |
| Comparison model | `meta-llama/Llama-3.2-3B-Instruct` | Small model, more corpus in view (Stage 1c comparison); not wired yet |
| Verifier | `cross-encoder/nli-deberta-v3-base` | A dedicated NLI model rather than an LLM judge |
| Verification unit | The citation group (`verifier.granularity: claim`) | Matches what each citation vouches for. Sentence-level verification is built behind the flag and measured; see the README |
| Partial answers | Off (`generator.partial_answers: false`) | The strict prompt refuses a whole question when any part is unsupported. The partial variant is built behind the flag and measured |
| Narrator | Local: reuses the loaded Phi-4 | No VRAM for a second model, and the demo runs offline. An API backend is an extension point, not wired |
| UI | Streamlit | The fastest way to a clean demo |
| τ (entailment threshold) | 0.5 | TBM |
| φ (min verified fraction) | 0.5 | TBM, and barely exercised: on the 27 probes every report that reached the gate had one claim; one conflict-fork report (k02) passed at exactly 1/2 |

## Open questions

- Which counterfactuals make the best "surprise" beat? They should be facts the model is very confident about.
- Should the poison fork be in the live demo or only in the slides? It is now one click, with no restart.
- Does the fabrication rate in `scripts/eval.py` match ENG 01's definition? It is a proposal (see the README's Evaluation section), and it has known false positives on declined text.
- Switch on sentence-level verification or partial answers? Both are measured side by side in the README; neither is on by default.
