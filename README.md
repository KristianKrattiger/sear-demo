# sear-demo

A small, honest demo of **SEAR** (Sparse Epistemically Anchored Reasoning), the constrained-inference layer of GIN.

> **Which SEAR this is.** This demo is SEAR's verify-after stage (GIN ENG 01, Stages 0–1): the model generates freely, then every claim is checked against the corpus and dropped if it fails. GIN's own repository also implements the decode-time stage, a logits processor that lets the model emit only spans copied from the corpus. Both are SEAR, but the mechanisms differ: this one catches fabrication after the fact, and that one makes it unreachable.

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
python scripts/eval.py --corpus corpus/conflict               # 5 probes on the conflict fork
python scripts/check_truncation.py                            # verifier input lengths, after an eval
python scripts/replay_repair.py p15                           # every citation-repair candidate for a dropped claim
python -m pytest -q                                           # 85 tests, no GPU or model downloads needed
```

`eval.py --out runs/<name>` writes the per-probe logs to another folder, and `--set section.key=value` runs a config flag on without editing the config (an unknown key is an error); `check_truncation.py --runs runs/<name>` reads such a folder. The first line of the eval's output names the commit it ran on, and says so when the working tree had uncommitted changes:

```bash
python scripts/eval.py --narrator --out runs/sentence --set verifier.granularity=sentence --set verifier.sentence_context=film
```

Opening the app, or starting an eval, loads all three models, which takes a while the first time. The app also builds the index itself if `index/chunks.faiss` is missing, and `eval.py` always rebuilds its own index under `index/<corpus folder name>/`, so editing chunks never leaves an eval on a stale index.

## Using the app

Pick a corpus, type a question and press Enter. The **Corpus** selector lists the main corpus and every fork under `corpus/` (poisoned, conflict). Switching swaps only the chunks and the index (`index/<folder>/`, rebuilt when missing or older than the chunks); the three models stay loaded, so it takes seconds. The page shows, top to bottom:

1. **Failure state**, if the generator refused or too few claims verified: "The corpus does not support an answer to this question." The grounded report is hidden; the retrieved chunks are still listed. With the narrator on, a caption says why it was skipped (the failure state, or fewer than `min_claims` verified sentences).
2. **Summary** (narrator on only). Each sentence ends with the handles it cites and, when a check fails, a badge. Its **Sources** expander shows each cited handle's verified sentence and the chunk or chunks that back that sentence. When no cited chunk, and no pair of them, entails the sentence on its own, the expander says so and lists every chunk the sentence's paragraph cited: the sentence was verified only as part of that paragraph.
3. **Grounded report**: every claim the generator made, marked ✅ or ✖️ dropped, with `(model cited ...)` when the verifier moved a wrong citation to the right chunk, the entailment score, any names or numbers the cited chunk lacks, and `checked as ...` when sentence-level verification scored the sentence with an antecedent added. Under each claim, its cited chunk IDs: click one and the chunk opens. With partial answers on, the parts the generator declined are listed under the claims, unverified.
4. **Retrieved chunks**: everything the generator was shown.

| Badge on a summary sentence | Meaning |
|---|---|
| none | Supported: entailed by the claims it cites, no contradiction, no new names or numbers |
| red "contradicts C2" | The cited claims together contradict it; the badge names the claims that conflict |
| orange "not in the cited claims: Best Animated Feature" | It names a person, title or number its cited claims never mention |
| orange "not supported by the report" | Not entailed by the claims it cites |
| orange "uncited" | It cites nothing, or only handles or chunk IDs the report doesn't have |

A warning under the summary lists any handles or chunk IDs the narrator cited that the report doesn't have.

The **Narrator** toggle starts at `narrator.enabled` from the config. Answers and summaries are cached per corpus and question for the browser session, so flipping the toggle never regenerates the grounded report: what you compare is the same answer with and without the summary.

**Compare corpora** asks one question on two corpora (the selected one and, by default, the poisoned fork) and shows the two grounded reports side by side: verdicts, entailment scores and clickable citations. The narrator is off in this view. On the poisoned fork the answers are fully cited, verified and wrong, which is the governance point.

Questions that show each behaviour (all from the probe sets):

| Behaviour | Question |
|---|---|
| Grounded answer, one cited claim | Who designed the creature in Alien? |
| Narrator demo (needs at least two verified sentences) | Give an overview of Spirited Away, including the awards it won. |
| Counterfactual: corpus beats memory | What year was Alien released? (corpus: 1983; memory: 1979) |
| Out of corpus: failure state | What was the budget of Alien? |
| Near miss: neighbouring facts, still no answer | Who plays Victor Laszlo in Casablanca? |
| Overreach: the source supports less than the natural answer | Who composed the full orchestral score of Jaws? |
| Poisoned fork: cited, verified, wrong | Who designed the creature in Alien? (main: Giger; poisoned: Druillet) |
| Conflict fork: two chunks disagree | What is the name of Quint's boat in Jaws? (Orca in one chunk, Pequod in another) |

`.streamlit/config.toml` turns Streamlit's file watcher off (it walked the `transformers` package on every rerun and logged about a hundred `torchvision` tracebacks), so restart the app after editing code or config. The models are shared by every browser session the app serves, and so is the corpus in use: the selector is meant for one presenter, not several people switching at once.

## Configuration

Everything is in `config/default.yaml`.

| Key | Default | What it does |
|---|---|---|
| `corpus.chunks` | `corpus/chunks.jsonl` | Chunks the app starts on and `build_index.py` loads (`eval.py --corpus` overrides it; the app's selector switches it) |
| `corpus.probes` | `corpus/probes.jsonl` | Probe set for `eval.py` (overridden by `--corpus`) |
| `corpus.index_dir` | `index/` | Where the FAISS index lives; the selector and `eval.py` use `index/<corpus folder>/` |
| `retrieval.embedder` | `BAAI/bge-m3` | Sentence-transformers model for retrieval (CPU); it embeds each chunk's `text` |
| `retrieval.top_k` | 12 | Chunks shown to the generator. With 13 chunks this is nearly the whole corpus; see Known limits |
| `generator.model` | `microsoft/phi-4` | Hugging Face causal LM, loaded whole on GPU 0. Fallback: `Qwen/Qwen3-8B` |
| `generator.compare_model` | `meta-llama/Llama-3.2-3B-Instruct` | Reserved for the small-model comparison; not wired yet |
| `generator.load_in_4bit` | true | NF4 4-bit quantisation via bitsandbytes |
| `generator.max_new_tokens` | 400 | Answer length cap; also caps the narrator, which reuses the generator |
| `generator.partial_answers` | false | true: the `GENERATE_PARTIAL` prompt answers what the sources support and declines the rest on a `NOT SUPPORTED: ...` line; declines are shown, not verified, and not counted by `phi`. See Results for its measured effect |
| `generator.log_attention` | false | Attention logging; the reducer in `attention.py` is a stub, and raw attentions don't fit in 12 GB |
| `verifier.model` | `cross-encoder/nli-deberta-v3-base` | NLI cross-encoder (CPU) |
| `verifier.tau` | 0.5 | Entailment a claim needs to verify |
| `verifier.repair_citations` | true | Re-cite failed claims to the retrieved chunks that do support them |
| `verifier.granularity` | `claim` | `claim` verifies each citation group as written (often a paragraph); `sentence` splits each group into sentences and verifies, keeps or drops each on its own, so `phi` counts sentences |
| `verifier.sentence_context` | `none` | Sentence mode only: what stands in for a split sentence's lost antecedent. `none`, `film` (prefix "This is about the film X."), `previous` (prefix the paragraph's previous sentence), `resolve` (replace a leading "It"/"He"/"She" with its antecedent). The sentence shown never changes |
| `gate.phi` | 0.5 | Minimum fraction of a report's claims that must verify, else the failure state |
| `gate.failure_message` | "The corpus does not support an answer to this question." | Shown for the failure state |
| `narrator.enabled` | false | Where the UI toggle starts, and the default for `Sear.ask()`. `eval.py` ignores it and uses `--narrator` |
| `narrator.backend` | `local` | `local` reuses the loaded generator; anything else raises at startup (the API backend is not wired) |
| `narrator.min_claims` | 2 | Skip the narrator when the report has fewer verified sentences than this |
| `narrator.contradict_tau` | 0.5 | Contradiction probability that badges a summary sentence |

`tau`, `phi` and `contradict_tau` are TBM ("to be measured"): sensible starting values, not tuned results. `phi` is practically unexercised: on the 27 probes and the poisoned fork every report that reached the gate had exactly one claim, and with one claim any `phi` gives the same verdict. The one report in the default runs where `phi` decided anything is the conflict fork's k02 (one of two claims verified: 1/2 = 0.5 passes).

## Evaluation

`scripts/eval.py` runs a probe set through the full pipeline and prints one line per metric. Probe layers are defined in [corpus/README.md](corpus/README.md); the main set has 27 probes (7 realism, 3 counterfactual, 4 multi-hop, 5 out-of-corpus, 4 near-miss, 4 overreach), the poisoned fork has 6 (5 poisoned, 1 out-of-corpus) and the conflict fork 5.

| Line | Pass means |
|---|---|
| `claims flagged unsupported by the gate` | Claims the verifier dropped, out of all claims the generator made |
| `claims re-cited by the verifier` | Claims whose citation the repair step moved to a supporting chunk |
| `answered correctly (<layer>)` | Not the failure state, and a verified claim contains the expected answer |
| `model followed corpus, raw (<layer>)` | The generator's raw text has the corpus value, before any checking |
| `refused (<layer>)` | Out-of-corpus and near-miss probes ended in the failure state |
| `overreach: model avoided it (raw)` / `nothing unsupported emitted` | The forbidden over-claim is absent from the raw text / from the verified claims. The raw line exempts generator refusals that echo the question; before commit f91cfbd it also exempted gate withholds, which counted p24's raw over-claim as avoided |
| `overreach: answered without the over-claim` | Not the failure state, and no verified claim has the over-claim. 0 under the strict prompt, which refuses them all |
| `overreach: gave the supportable part` | Same, and a verified claim has the part the corpus does support (`corpus/partial_expect.jsonl`; p27 has none) |
| `failure states` | All failure states, split into generator refusals ("NOT SUPPORTED") and gate withholds (claims made, too few verified), with the gate's probe IDs |
| `failure-state recall` | Probes that should refuse (out-of-corpus, near-miss) that did |
| `failure-state precision` | Failure states that were on probes that should refuse; the rest are listed as over-refusals |
| `claims per report` | How many claims each report that reached the gate had, and how many mixed verified and dropped claims. `phi` decides a verdict only for those |
| `fabrication rate, emitted` / `raw generator output` | A proposed definition (see below): answers that assert content the corpus lacks, out of probes that answered / out of all probes, judged on what the user sees / on the generator's output |
| `partial answers` | Reports that answered and declined a part (only with `generator.partial_answers`) |
| `conflict <probe>` | Conflict fork only, not scored: whether the answer states both conflicting values, picks one, or refuses; which value the raw output states; whether retrieval returned both chunks; what the claims cite |

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
| `source chains with no backing chunk or pair` | Cited handles whose sentence no cited chunk, and no pair of them, entails on its own: the Sources panel lists every chunk the paragraph cited and says so |
| `control: <type>` | One line per control type (the seed counts as `swap`; the others are in `narrator_controls.jsonl`): how many the narrator's labels badged and the gate withheld (check, then repair), which checks fired, and which slipped through |

**Fabrication rate (a proposed definition; check it against ENG 01).** An output fabricates when, judged without regard to which chunk it cites, it (1) names a person, title or number that no retrieved chunk mentions, (2) contains the probe's `prior`, the value the model knows from memory, when no retrieved chunk holds it, or (3) matches an overreach probe's `forbid_regex`. Checks 2 and 3 apply only to claims that carry a citation, because a refusal often echoes the question. The emitted rate judges the verified claims of reports that passed the gate (plus any declined text), over probes that answered; the raw rate judges the generator's output, over all probes. Limits: on emitted claims check 1 can never fire, since the gate already required every name and number to be in the cited chunks; and the judge can't see a false relation between names the corpus has ("Strasser shoots Rick") or an over-credit that no probe pattern describes (p08's verified "The music for Jaws was composed by Bernard Herrmann", where the corpus credits him with the two-note theme only).

Per-probe detail goes to `runs/<corpus>_eval.jsonl` (question, retrieved IDs, raw output, every claim with its citation, any correction, score, verdict, `checked` text and, for a claim repair couldn't fix, repair's best candidate; declined parts; fabrication reasons; conflict outcome) and `runs/<corpus>_narrator.jsonl` (handles, summary sentences with labels, scores and source chains, which chains fell back, the seed control and every typed control with the checks that fired). `runs/` is not tracked by git.

Decoding is greedy, so a rerun on the same commit and machine normally reproduces the same answers.

`scripts/check_truncation.py` rebuilds every NLI input the gate scored in the last eval and reports how many pass 512 tokens and whether the model receives them whole. See "Long inputs" in the architecture doc for what it found.

## Results

Every number here comes from a log in `runs/` (not tracked by git), written by `eval.py --out runs/<commit>-<config>`, whose first line names the commit. The four configurations ran at commit 8a165f6 (`runs/8a165f6-base`, `-partial`, `-sentence`, `-both`). Decoding is greedy: the 8a165f6 baseline reproduced the 6fdf05d baseline byte for byte, logs included.

### Four configurations

Commit 8a165f6. Partial answers use prompt v2 and sentence mode uses `sentence_context: previous` (see below for both). Where the four columns agree, one value is given.

**Main corpus** (27 probes, 4 narrator probes; `runs/8a165f6-<config>/corpus_stdout.txt`)

| Line | Baseline | Partial answers | Sentence mode | Both |
|---|---|---|---|---|
| answered correctly: realism, counterfactual, multi-hop | 7/7, 3/3, 3/4 | same | same | same |
| refused: out-of-corpus, near-miss | 5/5, 4/4 | same | same | same |
| overreach: nothing unsupported emitted | 4/4 | 4/4 | 4/4 | 4/4 |
| overreach: answered without the over-claim; gave the supportable part | 0/4; 0/3 | 0/4; 0/3 | 0/4; 0/3 | 0/4; 0/3 |
| overreach: model avoided it (raw) | 3/4 | 3/4 | 3/4 | 3/4 |
| failure states (generator refused, gate withheld) | 14/27 (11, 3: p15, p20, p24) | 14/27 (12, 2: p15, p24) | 14/27 (11, 3: p15, p20, p24) | 14/27 (12, 2: p15, p24) |
| failure-state recall; precision | 9/9; 9/14 | 9/9; 9/14 | 9/9; 9/14 | 9/9; 9/14 |
| answered and declined a part | 0/13 | 5/13 | 0/13 | 5/13 |
| claims flagged unsupported; re-cited | 3/16; 2/16 | 2/16; 2/16 | 3/16; 2/16 | 3/17; 2/17 |
| claims per report; mixed verdicts | 16 × 1; 0 | 14 × 1, 1 × 2; 0 | 16 × 1; 0 | 13 × 1, 2 × 2; 0 |
| fabrication rate: emitted; raw | 0/13; 2/27 (p20, p24) | 1/13 (p09); 3/27 (p09, p24, p27) | 0/13; 2/27 (p20, p24) | 1/13 (p09); 3/27 (p09, p24, p27) |
| narrator ran; summary sentences supported | 4/4; 14/14 | 4/4; 15/15 | 4/4; 14/14 | 4/4; 15/15 |
| narrator slips shown without a badge; kept the corpus value | 0/4; 4/4 | 0/4; 4/4 | 0/4; 4/4 | 0/4; 4/4 |
| source chains with no backing chunk or pair | 0/16 | 0/17 | 0/16 | 0/17 |
| seeded slip badged; gate withheld the seeded swap | 4/4; 4/4 | 4/4; 4/4 | 4/4; 4/4 | 4/4; 4/4 |
| all controls: narrator badged; gate withheld | 20/23; 18/23 | 18/21; 18/21 | 20/23; 18/23 | 18/21; 18/21 |

**Poisoned fork** (6 probes, 2 narrator probes)

| Line | Baseline | Partial answers | Sentence mode | Both |
|---|---|---|---|---|
| answered correctly (poisoned); refused (out-of-corpus) | 5/5; 1/1 | same | same | same |
| summary sentences supported; slips without a badge | 7/7; 0/2 | 5/5; 0/2 | 6/6; 0/2 | 5/5; 0/2 |
| source chains with no backing chunk or pair | 1/8 | 0/5 | 0/7 | 0/5 |
| all controls: narrator badged; gate withheld | 12/12; 12/12 | 9/9; 12/12 | 7/7; 12/12 | 9/9; 12/12 |

The conflict fork's outcomes are the same in all four configurations (below).

How to read the differences:

- **A control applies only when its text is in a verified handle of that run**, so the control counts change with the handles. The partial prompt and sentence mode change the generator's or the gate's output on the narrator probes, and some controls then had nothing to swap.
- **Sentence mode** changes nothing on the 27 probes, whose answers are single sentences, and nothing on the main corpus's narrator reports. On the poisoned fork it drops n06's "Quint is known for delivering a monologue … during their mission" (0.119), the one handle no chunk backed (1/8 becomes 0/7), and the five n06 controls written against that handle no longer apply.
- **Partial answers** produced no partial answer to an overreach probe (Partial answers, below). Its five "partial answers" are complete answers with a gratuitous decline, and its fabrication flags on p09 and p27 are judge false positives.
- **Both.** Sentence mode splits p24's v2 output into the over-claim and the supportable sentence ("The two-note theme that signals the shark's approach was also composed by him"). With `previous` the second is scored with the over-claim before it, so it is dropped too (0.000) and the overreach lines don't move. This is one cost of the `previous` remedy.
- **The candidate gate contradiction check** would have withheld 0 verified claims in all twelve runs (`runs/8a165f6-*/gate_contradiction.txt`).

### Controls by type

The six seeds swap in a name or number. The 29 typed controls (`corpus/narrator_controls.jsonl`, `corpus/poisoned/narrator_controls.jsonl`) write slips the names check can't see. Each is labelled like a narrator sentence, and in its source chunk's wording it goes through the gate's check and then repair. Main corpus and poisoned fork together, baseline run:

| Type | Controls | Narrator badged | Gate withheld | Which check fired | Slipped through |
|---|---|---|---|---|---|
| swap (the seeds) | 6 | 6/6 | 6/6 | names on all 6; entailment and contradiction on 5 (not the award swap, entailed at 0.991) | none |
| interpretation | 6 | 6/6 | 6/6 | entailment | none |
| over-credit | 7 | 5/7 | 5/7 | entailment (contradiction on 1) | n01 "written and directed by Ridley Scott" (narrator 0.996, gate 0.998); n02 "The music in Jaws was composed by Bernard Herrmann" (0.993, 0.992) |
| in-evidence swap | 6 | 5/6 | 5/6 | entailment and contradiction | n03 "written and directed by Studio Ghibli" (0.986 on both) |
| reversal (polarity or roles) | 6 | 6/6 | 5/6 | entailment and contradiction | gate: n01 "not directed by Ridley Scott" (0.978) |
| initial-name swap | 4 | 4/4 | 3/4 | narrator: names on 2 (Sam, Benchley), entailment and contradiction on 3; the two names the corpus never uses (Henreid, Williams) were caught by NLI alone | gate: n04 "Sam shoots the German Major Strasser", after repair |
| **total** | **35** | **32/35** | **30/35** | | |

Per control, with scores: `runs/8a165f6-base/<corpus>_narrator.jsonl` (`controls`).

What the slips show (from the logs, and the window-by-window scores in `runs/item2-diagnosis/slipped_controls_windows.txt`):

- **Contradiction never fired alone.** Every control the narrator's contradiction check caught had also fallen below `tau` on entailment. The names check alone caught two: the award swap (the gate and the narrator) and "Sam" (the narrator).
- **Over-credit and in-evidence swaps are NLI misses.** "written and directed by Ridley Scott" against "directed by Ridley Scott": entailment 0.996, contradiction 0.000. p08's real, verified answer makes the same over-credit: "The music for Jaws was composed by Bernard Herrmann, who created the two-note theme" (0.998), where the corpus credits him with the two-note theme only.
- **A window that omits a fact can entail its negation.** The gate scored "Alien is a science-fiction horror film not directed by Ridley Scott" 0.978 against the alien-c01 window about the screenplay, and 0.969 against the window "It follows the crew…". The windows that state "directed by Ridley Scott" contradict it at 1.000. The verifier keeps the best entailment over windows, so the claim passed. The narrator caught it: its only premise is the handle sentence, which names the director (entailment 0.000, contradiction 1.000).
- **Repair can rescue a slip.** The gate's check rejected "Sam shoots the German Major Strasser…" (Sam is not in casa-c02). Repair then re-cited it to casa-c02 + casa-c03: casa-c03 mentions "the pianist Sam", so the names check passed, and the casa-c02 window that says *Rick* shoots Strasser supplied the entailment (0.685). The names check and the entailment score read different evidence. On the f91cfbd run, before the names check looked at sentence-initial words, the same control passed without repair.

### Candidate checks, measured and not applied

Each would change what the gate passes, so neither is switched on. Both were replayed on the 8a165f6 baseline's logs (32 verified claims across the three corpora, 35 gate-side controls):

| Candidate | Script | Verified claims it would withhold | Slipped gate controls it would catch |
|---|---|---|---|
| Contradiction in the gate: the narrator's rule (cited chunks joined, `contradict_tau` 0.5) | `gate_contradiction.py` | 0/32 | 1 of 5: the polarity slip |
| Names in the entailing premise: one premise must both entail the claim and name everything in it | `names_in_premise.py` | 1/32: n06's poisoned paragraph, the one holding "... during their mission", which no chunk backs (0.425) | 2 of 5: the polarity slip (only the windows that name Ridley Scott count, and they contradict it) and the repaired "Sam" slip |

### Sentence-level verification: the antecedent remedies

On the 27 probes, sentence mode changes almost nothing, because every answer is one sentence: 16 reports with one sentence each. The one effect is that `film` drops p16, a correct multi-hop answer (multi-hop 2/4, precision 9/15; `runs/6fdf05d-sent-film`). `none`, `previous` and `resolve` match the baseline line for line. Antecedents get lost in the narrator probes' multi-sentence reports. `scripts/replay_sentences.py` re-verifies those reports in sentence mode, with the generator's own text and citations, under each remedy (commit 5d8f15c, on the 6fdf05d baseline's logs: `runs/5d8f15c-replay/`):

| Remedy | Main reports: sentences verified | Poisoned reports | Dropped |
|---|---|---|---|
| `none` | 15/16 | 8/9 | n03 "Her parents eat food meant for the spirits… prompting Chihiro…" (0.153; the lost antecedent); n05 "Sigourney Weaver plays Ellen Ripley… as she confronts the creature" (0.001; "confronts the creature" is in no chunk) |
| `film` | 16/16 | 8/9 | n05 Weaver sentence, as above |
| `previous` | 16/16 | 8/9 | n06 "Quint is known for delivering a monologue … during their mission" (0.119; "during their mission" is in no chunk) |
| `resolve` | 15/16 | 8/9 | n03 (scored as "Chihiro Ogino's parents…", 0.158) and n05, as for `none` |

In claim mode all 25 of those sentences were verified as parts of their paragraphs. Sentence mode drops at least one embellishment on the poisoned fork whichever remedy is used, but no remedy drops both: `previous` lets the Weaver sentence through on the strength of the sentence before it ("who encounter the creature"), and `none` lets "during their mission" through. The full sentence-mode runs use `previous`, the only remedy that loses neither a correct probe answer nor the antecedent sentence.

### Partial answers

`generator.partial_answers` was measured in two prompt versions. v1 (commit f050615, run `6fdf05d-partial`) was written before any run. v2 (8a165f6, run `8a165f6-partial`) was written after reading v1's outputs on these probes, so its overreach numbers are not a held-out measurement.

| Main corpus | strict (baseline) | partial v1 | partial v2 |
|---|---|---|---|
| overreach answered without the over-claim | 0/4 | 0/4 | 0/4 |
| overreach gave the supportable part | 0/3 | 0/3 | 0/3 |
| near misses refused | 4/4 | 4/4 | 4/4 |
| failure-state recall / precision | 9/9 / 9/14 | 9/9 / 9/14 | 9/9 / 9/14 |
| reports that answered and declined a part | 0/13 | 0/13 | 5/13 |

Neither version gave the supportable part of any overreach probe, so precision did not move. The near-miss line did not blur either, since nothing was answered with neighbouring facts. What the model did instead, from the logs:

- **v1** wrote the supportable part inside an uncited sentence about what the sources lack. p25: "The sources only mention that Spirited Away won the Academy Award for Best Picture, but do not specify any awards for Miyazaki himself as a director." The pipeline reads that as a decline, so nothing was left to verify. p24 made the same over-claim as under the strict prompt, and the gate withheld it.
- **v2** asks for the supported part as cited sentences. p24 now wrote "The full orchestral score of Jaws was composed by Bernard Herrmann. The two-note theme that signals the shark's approach was also composed by him [jaws-c03]": the over-claim and the supportable part as one cited claim, dropped as a whole. p25 kept the uncited "the sources only mention" sentence, and p26 and p27 refused. The five partial answers v2 did produce were complete answers with a gratuitous decline added, such as p05's "NOT SUPPORTED: The specific details of what happens to the suspects after this order."
- **Fabrication judge false positives.** The judge counted names echoed from the question in v1's declines (p11 "Aliens", p14 "English") until the question became evidence (commit 8a165f6). In v2 it flags p09's decline "Any other Academy Awards…" (the plural is not a whole-word match for the corpus's "Academy Award") and p27's decline line, which echoes the question with citations attached. So v2's lines read emitted 1/13 (p09) and raw 3/27 (p09, p24, p27). Read by hand, only p24's raw over-claim asserts something the corpus lacks; the other three flags are judge errors.

With sentence mode on as well, p24's two sentences are verified separately: see the four-configuration table.

### Conflict fork

Baseline, commit 8a165f6 (`runs/8a165f6-base/conflict_*`). Retrieval returned both conflicting chunks for every probe.

| Probe | Question | The two chunks say | Answer | Memory |
|---|---|---|---|---|
| k01 | What is the name of Quint's boat in Jaws? | Orca / Pequod | picked Orca, citing jaws-c02 | Orca |
| k02 | Who designed the creature in Alien? | Giger / Druillet | picked Giger. The raw answer stated both ("…influenced by … Philippe Druillet"), and the gate dropped the Druillet sentence (0.000), so the reader sees one value verified and the other marked dropped | Giger |
| k03 | Who shoots Major Strasser in Casablanca? | Rick / Ilsa | picked Rick. The model cited both chunks, and repair re-cited the claim to casa-c02 + casa-c01 | Rick |
| k04 | What year was Alien released? | 1983 (the counterfactual) / 1979 | picked 1979, citing the new chunk over the counterfactual | 1979 |
| k05 | Describe how Brody, Hooper and Quint hunt the shark in Jaws. | Orca / Pequod | both reach the reader, as two verified facts in separate sentences, with nothing to say they disagree | Orca |

In four of five answers the output silently picks one value, and every pick is the one the model knows from memory. Where both values reach the reader (k05), nothing marks them as a conflict. The pipeline has no detector for this, by design: the gate checks each claim against the chunks it cites, and each conflicting claim is supported by its own chunk. Partial answers, sentence mode and both give the same outcome on every conflict probe (`runs/8a165f6-*/conflict_stdout.txt`). With partial answers v1 (run `3de566c-partial`), k03 was verified citing casa-c04, the chunk that says Ilsa shoots Strasser (Known limits).

## Tests

`python -m pytest -q` runs 85 tests in about ten seconds. No model is downloaded: `tests/conftest.py` stubs `transformers` when it isn't installed, and the tests use fake NLI, generator and retriever objects. They cover segmentation, the gate, verification and citation repair (including repair's record of its best failed candidate), the names-and-numbers check (including sentence-initial names the corpus uses), the narrator's labels, handles, source chains and their fallbacks, the seed and typed controls, sentence mode and its antecedent remedies, partial answers and declines, the fabrication judge, the conflict-fork outcome, config overrides, corpus switching, and the Streamlit app run headless (`AppTest`, with a `Sear` that loads no model): the corpus selector, the per-corpus cache, the compare view and the citation popovers.

## Known limits

Numbers are from the 8a165f6 runs unless named otherwise; see Results.

- **Retrieval is not tested.** `top_k` 12 of 13 chunks means the generator sees almost the whole corpus, so refusals reflect the prompt and verifier, not retrieval. The conflict fork (17 chunks) is the first where retrieval leaves chunks out, and it still returned both conflicting chunks for every probe.
- **Failure-state precision is 9/14.** All four overreach probes and the p15 multi-hop end in the failure state: 5 of the 14 failure states are on probes that have a supportable answer. Recall is 9/9. The partial-answers flag was meant to fix this and did not: neither prompt version answered an overreach probe (Results).
- **p15 is a verifier miss.** The generator answered correctly ("Tom Skerritt plays Captain Dallas, the captain of the ship Nostromo…") and cited one chunk. Repair tried the right pair, alien-c02 + alien-c01 (candidate 13 of 33), which names everything, and the best entailment was 0.068: the claim needs the bridge "Captain Dallas is the Nostromo's captain", which neither chunk states. Recorded, not fixed.
- **Claims are paragraph-sized, and their sentences are supported by association.** Phi-4 usually cites a whole answer once. The narrator's handles are the sentences of a verified paragraph and inherit its verdict. In the narrator reports, two such sentences carry an embellishment no chunk states ("as she confronts the creature", "during their mission"). The Sources panel marks a handle no chunk or pair backs (1 of 8 cited handles on the poisoned fork, n06's "during their mission"). Sentence-level verification, behind a flag, drops one of the two, whichever antecedent remedy is used, and none drops both (Results).
- **`phi` is barely exercised.** On the 27 probes and the poisoned fork every report that reached the gate had one claim, so `phi` decided nothing. The only mixed report is the conflict fork's k02, which passed at exactly 1/2.
- **The NLI model has blind spots the other checks don't cover.**
  - It entails over-credits: "written and directed by Ridley Scott" from "directed by Ridley Scott" (0.998); "The music … was composed by Bernard Herrmann" from the two-note theme (0.992). p08's real verified answer has the same over-credit (0.998).
  - It entails a swap to a name that is already in the evidence: "written and directed by Studio Ghibli" (0.986).
  - It reads a window that omits a fact as entailing the fact's negation: "not directed by Ridley Scott" scored 0.978 against the window about the screenplay. The verifier keeps the best window score, so the gate passed it while the window naming the director contradicted it at 1.000.
  - It entailed a swapped award (0.991); the names-and-numbers check catches that one.
- **The names-and-numbers check is lexical.** It flags a fair paraphrase that renames something ("Oscar" for "Academy Award"). It can't see a swap that changes no name or number ("did not win", "Major Strasser shoots Rick"), or a swap to a name already in the cited evidence (Studio Ghibli, above). It skips a sentence-initial name the corpus never uses ("Henreid shoots…"; NLI caught both controls of this kind). And it reads all the cited chunks while the entailment comes from one window, so repair re-cited "Sam shoots the German Major Strasser" to a pair where one chunk names Sam and another, about Rick, supplied the entailment. The control passed the gate.
- **Repair picks the first citation that passes, not the best one.** On the conflict fork with partial answers v1 (run `3de566c-partial`), it re-cited "Rick Blaine shoots Major Strasser…" to casa-c04 + casa-c01 (entailment 0.58), and casa-c04 says Ilsa shoots him (contradiction 1.000 against casa-c04 alone, 0.62 against the pair). casa-c02 + casa-c01 would have passed honestly, and did in the baseline run. A gate contradiction check would have withheld it.
- **Two contradicting chunks are not detected.** In four of five conflict probes the answer silently states one value, the one the model knows from memory; in the fifth both values appear as separate facts. No check compares chunks with each other.
- **Long joined premises read poorly.** A fact stated word for word in one of the four Alien chunks scored 0.001 against all four joined. Scoring short windows and keeping the best recovers that recall, at the cost of the window problem above.
- **Verified is not true.** The poisoned fork passes the gate 5/5 with wrong facts. Counterfactual adherence (3/3) and poison vulnerability (5/5) are the same property, the answer follows the corpus: the check enforces the report, not the world.
- **The fabrication rate is a proposed definition.** On verified claims its names check can't fire, since the gate already demands every name be in the cited chunks; it can fire on declined text (partial answers), and there its two flags were false positives (a plural, and a question echoed with citations). It can't see a false relation between names the corpus has, or an over-credit no probe pattern describes.
- **Small sets.** 13 chunks, 27 + 6 + 5 probes, 6 narrator probes, 35 controls. Every number is a smoke test, not a measurement.
- **Not built yet:** attention logging (stub), the Llama comparison, the API narrator backend, any conflict detector, and the two candidate checks (Results), which are measured but not applied.

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
config/default.yaml        models, top-k, thresholds, behaviour flags, narrator switch
corpus/                    chunks.jsonl, probes.jsonl, narrator_probes.jsonl, narrator_controls.jsonl,
                           partial_expect.jsonl, poisoned/ and conflict/ forks, README (formats)
src/sear_demo/
  pipeline.py              Sear: wires the stages; answer(), narrate(), ask(), use_corpus()
  retrieve.py              BGE-M3 + FAISS retriever; use() switches corpus
  prompts.py               generation (strict and partial) and narrator prompts
  generate.py              local Hugging Face generator (4-bit, greedy, GPU 0)
  segment.py               citation-group segmentation; sentence splitter; sentence mode; partial-answer declines
  verify.py                NLI verifier, citation repair, names-and-numbers check, corpus names
  gate.py                  verified-fraction gate
  narrator.py              summary over the gated report, per-sentence labels, source chains, controls
  metrics.py               fabrication judge and conflict outcome (no model needed)
  schema.py                Chunk, Claim, Report, Narration, Result
  corpus.py, config.py     loaders, corpus discovery, config overrides
  attention.py             attention-grounding stub (Stage 1c)
app/app.py                 Streamlit UI
scripts/                   build_index.py, eval.py, check_truncation.py, replay_repair.py,
                           replay_sentences.py, gate_contradiction.py, names_in_premise.py
tests/                     pytest suite (no models needed)
docs/ARCHITECTURE.md       how it works and why
docs/DESIGN.md             demo script, default choices, open questions
```
