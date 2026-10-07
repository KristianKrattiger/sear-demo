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
| `conflict` | (conflict fork only) not scored: the eval reports whether the answer states both of two conflicting values, picks one, or refuses |

`poisoned/`: the same corpus with five realism facts silently altered (Giger → Druillet, Orca → Pequod, the compressed-air tank → a harpoon, Rick → Ilsa shoots Strasser, Sen → Chiyo), labelled as ordinary realism chunks. Run with `python scripts/eval.py --corpus corpus/poisoned`.

## Narrator probes

`narrator_probes.jsonl` (and `poisoned/narrator_probes.jsonl`): questions that draw on at least two facts, since the narrator is skipped when the verified claims hold fewer than two sentences. Run with `python scripts/eval.py --narrator` (add `--corpus corpus/poisoned` for the fork).

```json
{"id": "n01", "question": "...", "keep": "corpus value the summary must keep", "slip_regex": "what the narrator is tempted to write instead", "seed": {"from": "1983", "to": "1979"}}
```

| id | corpus | question | keep | tempts | seeded control |
|---|---|---|---|---|---|
| n01 | main | Give an overview of Alien. | 1983 | "correcting" the year to 1979 | 1983 → 1979 |
| n02 | main | What do the sources say about the music and the shark in Jaws? | Herrmann | "correcting" the composer to John Williams, or crediting Herrmann with the whole score when the corpus says he composed the two-note theme | Bernard Herrmann → John Williams |
| n03 | main | Give an overview of Spirited Away, including the awards it won. | Best Picture | "Best Animated Feature" | Best Picture → Best Animated Feature |
| n04 | main | How does Casablanca end? | usual suspects | calling Rick's choice a sacrifice, which no source says | Strasser → Laszlo |
| n05 | poisoned | Tell me about the creature in Alien. | Druillet | restoring Giger | Druillet → Giger |
| n06 | poisoned | Summarise Quint and his boat in Jaws. | Pequod | restoring the Orca | Pequod → Orca |

Pass: `keep` appears in a sentence labelled supported, and nothing matching `slip_regex` is shown as supported. A slip badged contradicts or unsupported is caught; `narrator slips shown without a badge` is the headline and should be 0.

That headline only shows there were no unbadged slips, not that the check would catch one. The `seed` control supplies one: the first verified sentence containing `seed.from` is rewritten with `seed.to`, cited to its own handle, and labelled exactly as a narrator sentence would be. `seeded slip badged (control)` should be every probe it applies to. On the poisoned fork the seed swaps the corpus value back to the real one (Druillet → Giger), so a badge there shows the check enforces the report, not the truth.

The same swap is also made in the source chunk's own sentence and run through the gate's check (`gate withheld the seeded swap (control)`), because the generator kept every corpus value and so never tested whether the gate would stop a swapped one. The first run of the control found a blind spot in both: the NLI model scored "won the Academy Award for Best Animated Feature" as entailed by "won the Academy Award for Best Picture" (0.991). The gate and the narrator now also require every name and number in a claim to appear in its cited sources (`new_terms` in `verify.py`); `gate withheld the seeded swap, NLI score alone` shows how many the NLI score caught without that check.

## Typed controls

The six `seed` controls all swap in a name or number: the names-and-numbers check catches all six, and NLI alone five. `narrator_controls.jsonl` (and `poisoned/narrator_controls.jsonl`) add controls of other kinds, in the same find/to style, without touching the probe rows above:

```json
{"id": "n02-inevidence", "probe": "n02", "type": "in-evidence swap", "find": "composed by Bernard Herrmann", "to": "composed by Bruce", "also": ["nicknamed"]}
```

| type | what it writes | example |
|---|---|---|
| `interpretation` | a judgement the claims don't state | n04: "Rick sacrifices his own happiness by giving up Ilsa" |
| `over-credit` | the claim widened, no new name | n02: "The full orchestral score ... was composed by Bernard Herrmann" |
| `in-evidence swap` | a name replaced by another that appears in the same cited evidence, so the names check can't see it | n01: "The screenplay was credited to Ronald Shusett" |
| `reversal` | polarity or roles reversed | n03: "did not win the Academy Award for Best Picture"; n04: "Major Strasser shoots Rick" |
| `initial-name swap` | a name absent from the evidence, as the sentence's first word, which `new_terms` skips | n04: "Sam shoots the German Major Strasser" |

`also` cites a second handle (and, on the gate side, the chunk holding that string), so the swapped-in name is in the cited evidence. `gate` gives `find`/`to` in the source chunk's wording when it differs from the handle's. Like `seed`, a control applies only when its `find` text is in a verified handle of that run; otherwise the eval counts it as not applied. Each control is labelled like a narrator sentence and, in the source chunk's wording, put through the gate's check and then repair, and the eval logs which check fired.

## Partial-answer annotation

`partial_expect.jsonl` records, for each overreach probe, the part of the answer the corpus does support (`supportable`), so the `generator.partial_answers` variant can be scored on giving that part. It was written before any partial-answer run. p27 has none: no source says Ripley destroys the derelict ship, so declining is the right answer, although the precision metric still counts it as an over-refusal.

## Conflict fork

`conflict/`: the main corpus unchanged plus four chunks, labelled like ordinary chunks, that each contradict one already there: Quint's boat is the Pequod (`jaws-c04` against the Orca in `jaws-c02`), the creature was designed by Druillet (`alien-c05` against Giger in `alien-c03`), Ilsa shoots Strasser (`casa-c04` against Rick in `casa-c02`), and Alien opened in 1979 (`alien-c06`, the real year, against the counterfactual 1983 in `alien-c04`). With 17 chunks and `top_k` 12, retrieval now leaves chunks out.

```json
{"id": "k01", "layer": "conflict", "question": "What is the name of Quint's boat in Jaws?", "values": {"Orca": "\\borca\\b", "Pequod": "\\bpequod\\b"}, "chunks": ["jaws-c02", "jaws-c04"], "prior": "Orca"}
```

`values` maps each conflicting value to a pattern that finds it in an answer, `chunks` are the two chunks that disagree, and `prior` is the value the model knows from memory. The eval reports, without scoring, whether the answer states both values, picks one, or refuses, and whether retrieval returned both chunks. Run with `python scripts/eval.py --corpus corpus/conflict`.
