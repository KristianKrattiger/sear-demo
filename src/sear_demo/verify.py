from transformers import pipeline

from .schema import Chunk, Claim


class Verifier:
    """NLI check: does the cited chunk entail the claim?"""

    def __init__(self, model: str, tau: float):
        self.nli = pipeline("text-classification", model=model, top_k=None)
        self.tau = tau

    def entailment(self, premise: str, hypothesis: str) -> float:
        scores = self.nli({"text": premise, "text_pair": hypothesis})
        return next((s["score"] for s in scores if s["label"].lower().startswith("entail")), 0.0)

    def check(self, claims: list[Claim], by_id: dict[str, Chunk]) -> list[Claim]:
        for claim in claims:
            sources = [by_id[i] for i in claim.cited if i in by_id]
            if not sources:
                claim.verified, claim.score = False, 0.0
                continue
            claim.score = max(self.entailment(s.text, claim.text) for s in sources)
            claim.verified = claim.score >= self.tau
        return claims
