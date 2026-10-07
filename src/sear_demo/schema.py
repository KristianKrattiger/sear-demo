from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    head: str = ""
    layer: str = "realism"
    meta: dict = field(default_factory=dict)


@dataclass
class Claim:
    text: str
    cited: list[str]
    verified: bool = False
    score: float = 0.0
    model_cited: list[str] | None = None  # set when the verifier corrected the model's citation
    label: str = ""  # narrator sentences only: supported | contradicts | unsupported | uncited
    contradicts: list[str] = field(default_factory=list)  # narrator sentences only: handles it contradicts
    new_terms: list[str] = field(default_factory=list)  # names/numbers its cited sources lack (any one fails it)
    repair_best: dict | None = None  # repair failed: its best candidate {cited, score, new_terms, tried}
    attention: dict[str, float] = field(default_factory=dict)  # chunk_id -> attention mass


@dataclass
class Report:
    query: str
    retrieved: list[Chunk]
    claims: list[Claim]
    failed: bool = False
    failure_message: str = ""
    raw: str = ""  # unprocessed generator output, for debugging

    @property
    def verified_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.verified]

    @property
    def cited_ids(self) -> set[str]:
        return {i for c in self.verified_claims for i in c.cited}

    @property
    def handles(self) -> dict[str, Claim]:
        """C1..Cn, one per sentence of the verified claims, in report order. The narrator cites these.

        The generator often cites a whole paragraph once, so one claim can hold six facts. Splitting
        gives the narrator atomic handles to cite and check against; each sentence keeps its claim's
        sources and score, since the gate verified the claim as a whole.
        """
        from .segment import split_sentences
        sents = [Claim(s, list(c.cited), verified=True, score=c.score)
                 for c in self.verified_claims for s in split_sentences(c.text)]
        return {f"C{i}": c for i, c in enumerate(sents, 1)}

    def chain(self, handles: list[str], sources: dict[str, list[str]] | None = None) -> list[tuple[str, Claim, list[Chunk]]]:
        """Resolve narrator handles to (handle, claim, retrieved chunks). `sources` (Narration.sources)
        narrows a handle to the chunks that back its sentence; otherwise it shows all its claim cited."""
        by_id = {c.id: c for c in self.retrieved}
        claims, sources = self.handles, sources or {}
        return [(h, claims[h], [by_id[i] for i in sources.get(h, claims[h].cited) if i in by_id])
                for h in handles if h in claims]


@dataclass
class Narration:
    sentences: list[Claim]  # reuses Claim: text, cited handles (C1..), score, verified, label, contradicts
    unknown_ids: list[str] = field(default_factory=list)  # handles or IDs the narrator cited that the report lacks
    skipped: str = ""  # why the narrator did not run; empty when it ran
    sources: dict[str, list[str]] = field(default_factory=dict)  # handle -> the chunk IDs that back that sentence


@dataclass
class Result:
    report: Report
    narration: Narration | None = None
