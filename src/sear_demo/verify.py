import logging
import re
from itertools import combinations

from transformers import pipeline

from .schema import Chunk, Claim
from .segment import split_sentences

NAME = re.compile(r"[A-Z][\w'\u2019-]*(?:\s+[A-Z][\w'\u2019-]*)*")
NUMBER = re.compile(r"\d(?:[\d,.]*\d)?")
POSSESSIVE = re.compile(r"['\u2019]s\b")
OPENERS = {"the", "a", "an", "its", "it", "his", "her", "their", "this", "that", "these", "those", "in", "on", "at",
           "after", "before", "when", "while", "as", "with", "by", "for", "from", "during", "although", "both",
           "each", "she", "he", "they", "there", "and", "but", "so", "then", "also", "despite", "once"}


def new_terms(text: str, evidence: str) -> list[str]:
    """Names and numbers in `text` that `evidence` never mentions.

    NLI reads "won Best Animated Feature" as entailed by "won Best Picture" (0.99 on the real model):
    the award changed but the sentence still sounds supported. A swapped name or number is the slip
    this misses, so the gate and the narrator also require every name and number to appear in the
    cited evidence. Names are runs of capitalised words; a lone capitalised word
    at the start of a sentence is skipped (it may just be capitalised there), and leading words like
    "The" are dropped. Matching ignores case and possessives.
    """
    pool = POSSESSIVE.sub("", evidence).lower()
    seen = lambda term: re.search(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)", pool) is not None
    terms = []
    for sent in split_sentences(text):
        start = re.search(r"\w", sent)
        for m in NAME.finditer(sent):
            for i, part in enumerate(POSSESSIVE.split(m.group())):
                words = part.split()
                if start and m.start() == start.start() and i == 0:
                    if len(words) == 1:
                        continue
                    while words and words[0].lower() in OPENERS:
                        words = words[1:]
                term = " ".join(words)
                if len(term) > 1 and not seen(term):
                    terms.append(term)
        terms += [n for n in NUMBER.findall(sent) if not seen(n)]
    return list(dict.fromkeys(terms))


class Verifier:
    """NLI check: does the cited chunk entail the claim, and does it name everything the claim names?"""

    def __init__(self, model: str, tau: float):
        self.nli = pipeline("text-classification", model=model, top_k=None, device="cpu")  # keep the GPU for the generator
        # A joint premise of four chunks runs past 512 tokens. DeBERTa-v3 uses relative positions only,
        # so longer inputs run without the indexing error the tokenizer warns about; truncating instead
        # would cut the last chunk off the premise. Silence the warning rather than truncate.
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
        self.tau = tau

    def scores(self, premise: str, hypothesis: str) -> dict[str, float]:
        """One NLI call: lowercased label (entailment, contradiction, neutral) -> probability."""
        return {s["label"].lower(): s["score"] for s in self.nli({"text": premise, "text_pair": hypothesis})}

    def entailment(self, premise: str, hypothesis: str) -> float:
        return next((v for k, v in self.scores(premise, hypothesis).items() if k.startswith("entail")), 0.0)

    def contradiction(self, premise: str, hypothesis: str) -> float:
        return next((v for k, v in self.scores(premise, hypothesis).items() if k.startswith("contradict")), 0.0)

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
                if score >= self.tau and not new_terms(claim.text, self.joint_premise(cand)):
                    claim.model_cited, claim.cited = claim.cited, [c.id for c in cand]
                    claim.score, claim.verified, claim.new_terms = score, True, []
                    break
        return claims

    def check(self, claims: list[Claim], by_id: dict[str, Chunk]) -> list[Claim]:
        for claim in claims:
            sources = [by_id[i] for i in claim.cited if i in by_id]
            if not sources:
                claim.verified, claim.score = False, 0.0
                continue
            claim.score = self.support(claim.text, sources)
            claim.new_terms = new_terms(claim.text, self.joint_premise(sources))
            claim.verified = claim.score >= self.tau and not claim.new_terms
        return claims
