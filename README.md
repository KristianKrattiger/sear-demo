# sear-demo

A small, honest demo of **SEAR** (Sparse Epistemically Anchored Reasoning), the constrained-inference layer of GIN.

It shows three things on a hand-built film corpus:

1. **Grounded answers.** Every claim cites a corpus chunk, and a verifier checks that the chunk actually supports it: an NLI model must find entailment, and every name and number in the claim must appear in the chunk (NLI alone read "won Best Animated Feature" as entailed by "won Best Picture").
2. **Corpus over memory.** On counterfactual probes, the answer follows the corpus even when the model "knows" otherwise.
3. **An explicit failure state.** When the corpus can't support an answer, SEAR says so instead of guessing.

An optional **narrator** (off by default) turns the dense grounded report into a readable summary. It sees only the verified report, one handle per verified sentence (C1, C2, ...), cites those handles, and each of its sentences is re-verified against the ones it cites, by the same two checks. The original grounded report is always shown underneath.

This is ENG 01 Stages 0–1 plus the Stage 4 narrator pulled forward. Nothing here is a measured spec yet (see the GIN ENG 00 promotion rule).

How it works inside, and why each check exists: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The demo script and default choices: [docs/DESIGN.md](docs/DESIGN.md). Corpus and probe formats: [corpus/README.md](corpus/README.md).

## Requirements

| | |
|---|---|
| Python | 3.10 or newer |
| GPU | NVIDIA, 12 GB (developed on an RTX 4070). Only the generator uses it: Phi-4 14B at 4-bit takes about 9 GB. |
| CPU | Runs the retrieval embedder (BGE-M3) and the NLI verifier, so the GPU stays free for the generator. |
| Disk | About 35 GB for Hugging Face model downloads, most of it Phi-4's bf16 weights (quantised to 4-bit when loaded). |
| Accounts | None. A Hugging Face login is only needed for the gated Llama 3.2 comparison model, which is not wired in yet. |

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
# 1. A CUDA build of PyTorch first: pick the command for your driver at https://pytorch.org/get-started/locally/
#    (the default pip wheel on Windows is CPU-only, and the generator refuses to fall back to CPU)
# 2. Then the package, with the 4-bit extras and pytest:
pip install -e ".[quant,dev]"
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

`quant` adds `bitsandbytes` and `accelerate`, which `generator.load_in_4bit: true` needs. `api` (the `anthropic` SDK) is listed for a future API narrator and is not used yet.

## Run

Run everything from the repository root; config and corpus paths are relative to it.

```bash
python scripts/build_index.py                                 # embed corpus/chunks.jsonl into index/chunks.faiss
streamlit run app/app.py                                      # the demo UI
python scripts/eval.py --narrator                             # 27 probes + 4 narrator probes on corpus/
python scripts/eval.py --corpus corpus/poisoned --narrator    # 6 probes + 2 narrator probes on the poisoned fork
python scripts/check_truncation.py                            # verifier input lengths, after an eval
python -m pytest -q                                           # 41 tests, no GPU or model downloads needed
```

Opening the app, or starting an eval, loads all three models, which takes a while the first time. The app also builds the index itself if `index/chunks.faiss` is missing, and `eval.py` always rebuilds its own index under `index/<corpus folder name>/`, so editing chunks never leaves an eval on a stale index.

## Using the app

Type a question and press Enter. The page shows, top to bottom:

1. **Failure state**, if the generator refused or too few claims verified: "The corpus does not support an answer to this question." The grounded report is hidden; the retrieved chunks are still listed. With the narrator on, a caption says why it was skipped (the failure state, or fewer than `min_claims` verified sentences).
2. **Summary** (narrator on only). Each sentence ends with the handles it cites and, when a check fails, a badge. Its **Sources** expander shows each cited handle's verified sentence and the chunk or chunks that back that sentence.
3. **Grounded report**: every claim the generator made, marked ✅ or ✖️ dropped, with its cited chunk IDs, `(model cited ...)` when the verifier moved a wrong citation to the right chunk, the entailment score, and any names or numbers the cited chunk lacks.
4. **Retrieved chunks**: everything the generator was shown.

| Badge on a summary sentence | Meaning |
|---|---|
| none | Supported: entailed by the claims it cites, no contradiction, no new names or numbers |
| red "contradicts C2" | The cited claims together contradict it; the badge names the claims that conflict |
| orange "not in the cited claims: Best Animated Feature" | It names a person, title or number its cited claims never mention |
| orange "not supported by the report" | Not entailed by the claims it cites |
| orange "uncited" | It cites nothing, or only handles or chunk IDs the report doesn't have |

A warning under the summary lists any handles or chunk IDs the narrator cited that the report doesn't have.

The **Narrator** toggle starts at `narrator.enabled` from the config. Answers and summaries are cached per question for the browser session, so flipping the toggle never regenerates the grounded report: what you compare is the same answer with and without the summary.

Questions that show each behaviour (all from the probe sets):

| Behaviour | Question |
|---|---|
| Grounded answer, narrator demo | Give an overview of Spirited Away, including the awards it won. |
| Counterfactual: corpus beats memory | What year was Alien released? (corpus: 1983; memory: 1979) |
| Out of corpus: failure state | What was the budget of Alien? |
| Near miss: neighbouring facts, still no answer | Who plays Victor Laszlo in Casablanca? |
| Overreach: the source supports less than the natural answer | Who composed the full orchestral score of Jaws? |

To show the **poisoned fork** in the UI, set `corpus.chunks: corpus/poisoned/chunks.jsonl` and `corpus.index_dir: index/poisoned/` in `config/default.yaml` and restart the app. Its answers are fully cited, verified and wrong, which is the governance point.

`.streamlit/config.toml` turns Streamlit's file watcher off (it walked the `transformers` package on every rerun and logged about a hundred `torchvision` tracebacks), so restart the app after editing code or config.

## Configuration

Everything is in `config/default.yaml`.

| Key | Default | What it does |
|---|---|---|
| `corpus.chunks` | `corpus/chunks.jsonl` | Chunks the app and `build_index.py` load (`eval.py --corpus` overrides it) |
| `corpus.probes` | `corpus/probes.jsonl` | Probe set for `eval.py` (overridden by `--corpus`) |
| `corpus.index_dir` | `index/` | Where the FAISS index lives |
| `retrieval.embedder` | `BAAI/bge-m3` | Sentence-transformers model for retrieval (CPU); it embeds each chunk's `text` |
| `retrieval.top_k` | 12 | Chunks shown to the generator. With 13 chunks this is nearly the whole corpus; see Known limits |
| `generator.model` | `microsoft/phi-4` | Hugging Face causal LM, loaded whole on GPU 0. Fallback: `Qwen/Qwen3-8B` |
| `generator.compare_model` | `meta-llama/Llama-3.2-3B-Instruct` | Reserved for the small-model comparison; not wired yet |
| `generator.load_in_4bit` | true | NF4 4-bit quantisation via bitsandbytes |
| `generator.max_new_tokens` | 400 | Answer length cap; also caps the narrator, which reuses the generator |
| `generator.log_attention` | false | Attention logging; the reducer in `attention.py` is a stub, and raw attentions don't fit in 12 GB |
| `verifier.model` | `cross-encoder/nli-deberta-v3-base` | NLI cross-encoder (CPU) |
| `verifier.tau` | 0.5 | Entailment a claim needs to verify |
| `verifier.repair_citations` | true | Re-cite failed claims to the retrieved chunks that do support them |
| `gate.phi` | 0.5 | Minimum fraction of a report's claims that must verify, else the failure state |
| `gate.failure_message` | "The corpus does not support an answer to this question." | Shown for the failure state |
| `narrator.enabled` | false | Where the UI toggle starts, and the default for `Sear.ask()`. `eval.py` ignores it and uses `--narrator` |
| `narrator.backend` | `local` | `local` reuses the loaded generator; anything else raises at startup (the API backend is not wired) |
| `narrator.min_claims` | 2 | Skip the narrator when the report has fewer verified sentences than this |
| `narrator.contradict_tau` | 0.5 | Contradiction probability that badges a summary sentence |

`tau`, `phi` and `contradict_tau` are TBM ("to be measured"): sensible starting values, not tuned results.

## Evaluation

`scripts/eval.py` runs a probe set through the full pipeline and prints one line per metric. Probe layers are defined in [corpus/README.md](corpus/README.md); the main set has 27 probes (7 realism, 3 counterfactual, 4 multi-hop, 5 out-of-corpus, 4 near-miss, 4 overreach) and the poisoned fork has 6 (5 poisoned, 1 out-of-corpus).

| Line | Pass means |
|---|---|
| `claims flagged unsupported by the gate` | Claims the verifier dropped, out of all claims the generator made |
| `claims re-cited by the verifier` | Claims whose citation the repair step moved to a supporting chunk |
| `answered correctly (<layer>)` | Not the failure state, and a verified claim contains the expected answer |
| `model followed corpus, raw (<layer>)` | The generator's raw text has the corpus value, before any checking |
| `refused (<layer>)` | Out-of-corpus and near-miss probes ended in the failure state |
| `overreach: model avoided it (raw)` / `nothing unsupported emitted` | The forbidden over-claim is absent from the raw text / from the verified claims |
| `failure states` | All failure states, split into generator refusals ("NOT SUPPORTED") and gate withholds (claims made, too few verified), with the gate's probe IDs |
| `failure-state recall` | Probes that should refuse (out-of-corpus, near-miss) that did |
| `failure-state precision` | Failure states that were on probes that should refuse; the rest are listed as over-refusals |

With `--narrator`, the narrator probes add:

| Line | Pass means |
|---|---|
| `narrator slips shown without a badge` | The headline: a tempting slip shown as supported. Should be 0 |
| `summary sentences supported` | Summary sentences that passed every check |
| `summary sentences uncited or unknown handle` | Sentences citing nothing usable |
| `narrator ran` | The report had at least `min_claims` verified sentences |
| `narrator kept the corpus value (runs)` / `avoided the slip, raw (runs)` | Scored only on probes where the narrator ran |
| `seeded slip badged (control)` | A known slip, written into a verified sentence on purpose, gets a badge. Should be every probe |
| `gate withheld the seeded swap (control)` | The same swap in the source chunk's sentence, run through the gate's check, is withheld |
| `gate withheld the seeded swap, NLI score alone` | How many of those the entailment score caught without the names-and-numbers check |

Per-probe detail goes to `runs/<corpus>_eval.jsonl` (question, retrieved IDs, raw output, every claim with its citation, any correction, score and verdict) and `runs/<corpus>_narrator.jsonl` (handles, summary sentences with labels, scores and source chains, and both controls). `runs/` is not tracked by git.

Decoding is greedy, so a rerun on the same commit and machine normally reproduces the same answers.

`scripts/check_truncation.py` rebuilds every NLI input the gate scored in the last eval and reports how many pass 512 tokens and whether the model receives them whole. See "Long inputs" in the architecture doc for what it found.

## Tests

`python -m pytest -q` runs 41 tests in under a second. No model is downloaded: `tests/conftest.py` stubs `transformers` when it isn't installed, and the tests use fake NLI, generator and retriever objects. They cover segmentation, the gate, verification and citation repair, the names-and-numbers check, the narrator's labels, handles and source chains, the seeded control, and the pipeline's narrator wiring.

## Known limits

- **Retrieval is not tested.** `top_k` 12 of 13 chunks means the generator sees almost the whole corpus, so refusals reflect the prompt and verifier, not retrieval.
- **Claims are paragraph-sized.** Phi-4 usually cites a whole answer once, so the gate verifies paragraphs and the main eval has few claims (16 on the last run). The narrator splits verified claims into one handle per sentence, but the gate's verdict is still per paragraph.
- **The NLI model has blind spots.** It scored a swapped award as entailed (0.991), which the names-and-numbers check now catches. That check is lexical: it flags a fair paraphrase that renames something ("Oscar" for "Academy Award"), and it can't see a swap that changes no name or number ("won" for "lost"), which is left to NLI contradiction.
- **Long joined premises read poorly.** A fact stated word for word in one of the four Alien chunks scored 0.001 against all four joined. The verifier scores short windows and takes the best, so this costs recall on multi-chunk claims rather than letting bad claims through.
- **Verified is not true.** The poisoned fork passes the gate 5/5 with wrong facts. The gate enforces the corpus, not the world.
- **Small sets.** 13 chunks, 27 + 6 probes, 6 narrator probes. Every number is a smoke test, not a measurement.
- **Not built yet:** attention logging (stub), the Llama comparison, the API narrator backend.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is False | You have the CPU wheel. Reinstall PyTorch with the CUDA command from pytorch.org |
| CUDA out of memory while loading | Close other GPU programs. The generator loads only on GPU 0 and fails instead of offloading. Fallback: `generator.model: Qwen/Qwen3-8B` |
| Error about bitsandbytes | `pip install -e ".[quant]"`. Turning `load_in_4bit` off is not an option on 12 GB: Phi-4 unquantised needs about 28 GB |
| `Token indices sequence length is longer than ... (512)` | Silenced on purpose: the verifier does not truncate and DeBERTa-v3 handles longer inputs. See the architecture doc |
| Hundreds of `torchvision` tracebacks from Streamlit | Run from the repo root so `.streamlit/config.toml` is picked up |
| App answers from old chunks after editing the corpus | `python scripts/build_index.py`, then restart the app |

## Layout

```
config/default.yaml        models, top-k, thresholds, narrator switch
corpus/                    chunks.jsonl, probes.jsonl, narrator_probes.jsonl, poisoned/ fork, README (formats)
src/sear_demo/
  pipeline.py              Sear: wires the stages; answer(), narrate(), ask()
  retrieve.py              BGE-M3 + FAISS retriever
  prompts.py               generation and narrator prompts
  generate.py              local Hugging Face generator (4-bit, greedy, GPU 0)
  segment.py               citation-group segmentation; sentence splitter
  verify.py                NLI verifier, citation repair, names-and-numbers check
  gate.py                  verified-fraction gate
  narrator.py              summary over the gated report, per-sentence labels, source chains
  schema.py                Chunk, Claim, Report, Narration, Result
  corpus.py, config.py     loaders
  attention.py             attention-grounding stub (Stage 1c)
app/app.py                 Streamlit UI
scripts/                   build_index.py, eval.py, check_truncation.py
tests/                     pytest suite (no models needed)
docs/ARCHITECTURE.md       how it works and why
docs/DESIGN.md             demo script, default choices, open questions
```
