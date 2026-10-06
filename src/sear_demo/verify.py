import re

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

    def check(self, claims: list[Claim], by_id: dict[str, Chunk]) -> list[Claim]:
        for claim in claims:
            sources = [by_id[i] for i in claim.cited if i in by_id]
            if not sources:
                claim.verified, claim.score = False, 0.0
                continue
            claim.score = max(self.entailment(p, claim.text) for s in sources for p in self.premises(s))
            claim.verified = claim.score >= self.tau
        return claims
