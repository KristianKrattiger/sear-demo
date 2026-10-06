import re

from .schema import Claim

CITE_GROUP = re.compile(r"(?:\s*\[[^\[\]]+\])+")
CITE = re.compile(r"\[([^\[\]]+)\]")


def _clean(text: str) -> str:
    text = re.sub(r"\s+([.!?,;])", r"\1", text).strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def segment(text: str) -> list[Claim]:
    """Claims are the spans that end in a citation group like [a] or [a, b][c].

    Splitting on citations rather than sentence punctuation keeps initials
    ("H. R. Giger") and abbreviations inside one claim. Trailing text with no
    citation becomes an uncited claim, which the verifier will reject.
    """
    claims, pos = [], 0
    for m in CITE_GROUP.finditer(text):
        body = text[pos:m.start()].strip()
        cited = [i.strip() for g in CITE.findall(m.group(0)) for i in g.split(",") if i.strip()]
        pos = m.end()
        while pos < len(text) and text[pos] in ".!?;":
            pos += 1
        if body:
            claims.append(Claim(text=_clean(body), cited=cited))
    tail = text[pos:].strip()
    if tail:
        claims.append(Claim(text=_clean(tail), cited=[]))
    return claims
