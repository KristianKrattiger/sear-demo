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


@dataclass
class Narration:
    sentences: list[Claim]  # reuses Claim: text, cited IDs, verified flag
    unknown_ids: list[str] = field(default_factory=list)


@dataclass
class Result:
    report: Report
    narration: Narration | None = None
