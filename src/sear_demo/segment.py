import re

from .schema import Claim

CITE = re.compile(r"\[([A-Za-z0-9_\-]+(?:\s*,\s*[A-Za-z0-9_\-]+)*)\]")
SENT = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z])")


def segment(text: str) -> list[Claim]:
    """Sentence-level claims with their cited IDs. Crude by design (ENG 01 §4)."""
    claims = []
    for sent in SENT.split(text.strip()):
        sent = sent.strip()
        if not sent:
            continue
        cited = [i.strip() for m in CITE.findall(sent) for i in m.split(",")]
        bare = re.sub(r"\s+([.!?,;])", r"\1", CITE.sub("", sent)).strip()
        if bare:
            claims.append(Claim(text=bare, cited=cited))
    return claims
