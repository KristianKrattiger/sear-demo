import re

from .schema import Claim

CITE_GROUP = re.compile(r"(?:\s*\[[^\[\]]+\])+")
CITE = re.compile(r"\[([^\[\]]+)\]")
SENT_END = re.compile(r"""[.!?]["'\u201d\u2019)\]]*(?=\s+["'\u201c\u2018(\[]?[A-Z0-9])""")
NOT_END = re.compile(r"(?:^|[\s(.])(?:[A-Z]|Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|Mt|Prof)\.$")


def _clean(text: str) -> str:
    text = re.sub(r"\s+([.!?,;])", r"\1", text).strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def split_sentences(text: str) -> list[str]:
    """Sentences of a claim. Initials ("H. R. Giger"), titles ("Dr.") and "U.S." stay inside one sentence;
    a closing quote after the full stop stays with the sentence it ends."""
    out, start = [], 0
    for m in SENT_END.finditer(text):
        if m.group(0)[0] == "." and NOT_END.search(text[start:m.start() + 1]):
            continue
        out.append(text[start:m.end()].strip())
        start = m.end()
    out.append(text[start:].strip())
    return [s for s in out if s]


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
