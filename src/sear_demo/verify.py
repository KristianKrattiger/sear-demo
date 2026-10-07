import re
from itertools import combinations

from transformers import pipeline

from .schema import Chunk, Claim


class Verifier:
    """NLI check: does the cited chunk entail the claim?"""

    def __init__(self, model: str, tau: float):
        self.nli = pipeline("text-classification", model=model, top_k=None, device="cpu")  # keep the GPU for the generator
        self.tau = tau

    def entailment(self, premise: str, hypothesis: str) -> float:
        scores = self.nli({"text": premise, "text_pair": hypothesis})
        return next((s["score"] for s in scores if s["label"].lower().startswith("entail")), 0.0)

    @staticmethod
    def premises(chunk: Chunk) -> list[str]:
        """NLI models are trained on short premises, so score the claim against
        windows of 1-3 sentences as well as the whole chunk, each prefixed with
        the film title (chunks often say "the film" rather than its name)."""
        sents = [s for s in re.split(r"(?<=[.!?])\s+", chunk.text) if s]
        windows = {chunk.text}
        for n in (1, 2, 3):
            windows.update(" ".join(sents[i:i + n]) for i in range(len(sents) - n + 1))
        film = chunk.meta.get("film")
        return [f"This is about the film {film}. {w}" if film else w for w in windows]

    @staticmethod
    def joint_premise(chunks: list[Chunk]) -> str:
        """All cited chunks together, each with its film prefix."""
        return " ".join(
            f"This is about the film {c.meta['film']}. {c.text}" if c.meta.get("film") else c.text
            for c in chunks
        )

    @classmethod
    def focused_joint_premise(cls, chunks: list[Chunk], claim: str) -> str:
        """The window of each chunk that shares the most words with the claim, joined.

        Two whole chunks make a long premise that the NLI model reads poorly; the
        two sentences that carry each hop are what it needs to see side by side.
        """
        words = lambda t: set(re.findall(r"\w+", t.lower()))
        target = words(claim)
        return " ".join(max(cls.premises(c), key=lambda w: (len(words(w) & target), -len(w))) for c in chunks)

    def support(self, claim: str, sources: list[Chunk]) -> float:
        premises = [p for s in sources for p in self.premises(s)]
        if len(sources) > 1:
            # a two-hop claim joins facts from several chunks, so no single chunk entails it
            premises.append(self.joint_premise(sources))
            premises.append(self.focused_joint_premise(sources, claim))
        return max(self.entailment(p, claim) for p in premises)

    def repair(self, claims: list[Claim], retrieved: list[Chunk], pair_pool: int = 6) -> list[Claim]:
        """Re-cite claims that failed against the chunks the model named.

        The model sometimes states a supported fact but cites the wrong chunk, or
        only one of the two chunks a two-hop answer needs. Search the retrieved
        chunks for a citation that does entail the claim: single chunks first,
        then the model's own citations plus one more chunk, then pairs from the
        top `pair_pool`. A repaired claim keeps the model's original citation in
        `model_cited` so the UI can show the correction. Uncited text and claims
        no retrieved chunk supports stay unverified.
        """
        by_id = {c.id: c for c in retrieved}
        for claim in claims:
            if claim.verified or not claim.cited:
                continue
            own = [by_id[i] for i in claim.cited if i in by_id]
            candidates = [[c] for c in retrieved]
            candidates += [own + [c] for c in retrieved if c not in own]
            candidates += [list(p) for p in combinations(retrieved[:pair_pool], 2)]
            seen = set()
            for cand in candidates:
                key = frozenset(c.id for c in cand)
                if key in seen:
                    continue
                seen.add(key)
                score = self.support(claim.text, cand)
                if score >= self.tau:
                    claim.model_cited, claim.cited = claim.cited, [c.id for c in cand]
                    claim.score, claim.verified = score, True
                    break
        return claims

    def check(self, claims: list[Claim], by_id: dict[str, Chunk]) -> list[Claim]:
        for claim in claims:
            sources = [by_id[i] for i in claim.cited if i in by_id]
            if not sources:
                claim.verified, claim.score = False, 0.0
                continue
            claim.score = self.support(claim.text, sources)
            claim.verified = claim.score >= self.tau
        return claims
