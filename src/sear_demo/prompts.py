GENERATE = """Answer the question using ONLY the sources below.
Every sentence must end with one or more source IDs in square brackets, like [f0001-c03].
If a sentence combines facts from more than one source, cite every source it uses, like [f0001-c01][f0001-c02].
If the sources do not answer the question, reply exactly: NOT SUPPORTED
In that case write nothing else: do not guess, and do not name people, dates or facts the sources do not contain.

Sources:
{sources}

Question: {query}
Answer:"""

NARRATE = """Below is a verified report. Each claim is followed by the source IDs that support it.
Write a short, readable summary of the findings for a non-specialist.
Rules: use only what the report says; end every sentence with the source IDs it relies on, copied from the report, like [f0001-c03]; do not add new facts.

Report:
{report}

Summary:"""


def format_sources(chunks) -> str:
    return "\n".join(f"[{c.id}] {c.text}" for c in chunks)


def format_report(report) -> str:
    return "\n".join(f"- {c.text}" for c in report.verified_claims)
