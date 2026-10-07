import re

from .schema import Claim

CITE_GROUP = re.compile(r"(?:\s*\[[^\[\]]+\])+")
CITE = re.compile(r"\[([^\[\]]+)\]")
SENT_END = re.compile(r"""[.!?]["'\u201d\u2019)\]]*(?=\s+["'\u201c\u2018(\[]?[A-Z0-9])""")
NOT_END = re.compile(r"(?:^|[\s(.])(?:[A-Z]|Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|Mt|Prof)\.$")
NAME = re.compile(r"[A-Z][\w'\u2019-]*(?:\s+[A-Z][\w'\u2019-]*)*")
# words that start a sentence capitalised without being a name ("The Nostromo", "After Kane ...")
OPENERS = {"the", "a", "an", "its", "it", "his", "her", "their", "this", "that", "these", "those", "in", "on", "at",
           "after", "before", "when", "while", "as", "with", "by", "for", "from", "during", "although", "both",
           "each", "she", "he", "they", "there", "and", "but", "so", "then", "also", "despite", "once"}
LEADING_REFERENCE = re.compile(r"^(The film|This film|The movie|It|Its|He|His|She|Her|They|Their)\b")


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


def first_name(sentence: str) -> str | None:
    """The first name in a sentence, after dropping capitalised openers ("The", "After") at its start.
    Unlike new_terms, a lone word at the start counts: "Rick shoots Strasser." is about Rick."""
    for m in NAME.finditer(sentence):
        words = m.group().split()
        while m.start() == 0 and words and words[0].lower() in OPENERS:
            words = words[1:]
        if words:
            return re.sub(r"['’]s$", "", " ".join(words))
    return None


def resolve_reference(sentence: str, previous: str, film: str | None) -> str:
    """Replace a leading "It"/"The film" with the film, or "He"/"She"/"They" with the first name of
    the previous sentence, keeping possessives ("Its" -> "Alien's"). Unchanged when there is none."""
    m = LEADING_REFERENCE.match(sentence)
    if not m:
        return sentence
    word = m.group(1)
    who = film if word in ("The film", "This film", "The movie", "It", "Its") else first_name(previous) if previous else None
    if not who:
        return sentence
    return who + ("'s" if word in ("Its", "His", "Her", "Their") else "") + sentence[m.end():]


def sentence_claims(claims: list[Claim], context: str = "none", films: dict[str, str] | None = None) -> list[Claim]:
    """One claim per sentence, each keeping its claim's citations, so each sentence is verified, kept
    or dropped on its own (verifier.granularity: sentence) and phi counts sentences.

    A split sentence can lose its antecedent ("It was released in 1983."). `context` sets what the
    verifier scores in its place, in `Claim.checked`; the sentence shown is never changed:
      none      the sentence as written
      film      "This is about the film X." before it, X the film of the claim's first cited chunk
      previous  the claim's previous sentence before it, so the two must be supported together
      resolve   a leading pronoun or "The film" replaced by its antecedent (resolve_reference)
    """
    if context not in ("none", "film", "previous", "resolve"):
        raise ValueError(f"unknown sentence_context {context!r}")
    films, out = films or {}, []
    for claim in claims:
        film = next((films[i] for i in claim.cited if films.get(i)), None)
        sents = split_sentences(claim.text) or [claim.text]
        for i, s in enumerate(sents):
            prev = sents[i - 1] if i else ""
            checked = {"none": s, "film": f"This is about the film {film}. {s}" if film else s,
                       "previous": f"{prev} {s}".strip(), "resolve": resolve_reference(s, prev, film)}[context]
            out.append(Claim(s, list(claim.cited), checked=checked if checked != s else ""))
    return out
