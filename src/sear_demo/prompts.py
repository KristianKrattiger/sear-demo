GENERATE = """Answer the question using ONLY the sources below.
Every sentence must end with one or more source IDs in square brackets, like [f0001-c03].
If a sentence combines facts from more than one source, cite every source it uses, like [f0001-c01][f0001-c02].
If the sources do not answer the question, reply exactly: NOT SUPPORTED
In that case write nothing else: do not guess, and do not name people, dates or facts the sources do not contain.

Sources:
{sources}

Question: {query}
Answer:"""

NARRATE = """You are summarising a verified report for a non-specialist.
The report answers the question below. Each claim has a handle in square brackets.

Rules:
- Use only what the claims say. Do not add facts, causes, opinions or background.
- Keep every name, date and number exactly as the claims give them, even if you believe they are wrong.
- End every sentence with the handles of the claims it uses, like [C1] or [C1][C2].
- Write 2 to 4 sentences. No lists, no headings.

Question: {query}

Report:
{report}

Summary:"""


def format_sources(chunks) -> str:
    return "\n".join(f"[{c.id}] {c.text}" for c in chunks)


def format_report(report) -> str:
    """The narrator's whole view of the world: verified sentences and their handles, no chunk IDs."""
    return "\n".join(f"[{h}] {c.text}" for h, c in report.handles.items())
