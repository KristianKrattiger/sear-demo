import streamlit as st

from sear_demo.config import load_config
from sear_demo.corpus import corpora
from sear_demo.pipeline import Sear

st.set_page_config(page_title="SEAR demo", layout="wide")


@st.cache_resource
def get_sear():
    return Sear(load_config())


def badge(s):
    if s.label == "contradicts":
        return f" :red[contradicts {', '.join(s.contradicts)}]"
    if s.label == "unsupported" and s.new_terms:
        return f" :orange[not in the cited claims: {', '.join(s.new_terms)}]"
    return {"unsupported": " :orange[not supported by the report]", "uncited": " :orange[uncited]"}.get(s.label, "")


def citations(ids, report, key):
    """Each cited chunk ID as a link that opens the chunk."""
    by_id = {c.id: c for c in report.retrieved}
    with st.container(horizontal=True, gap="small"):
        for i in ids:
            with st.popover(i, type="tertiary", key=f"{key}-{i}"):
                st.markdown(by_id[i].text if i in by_id else "Not among the retrieved chunks.")


def grounded_report(report, key):
    """Every claim with its verdict, entailment score and citations (none when the report failed), then
    anything the generator declined (partial answers)."""
    for n, c in enumerate([] if report.failed else report.claims):
        mark = "✅" if c.verified else "✖️ dropped"
        fix = f" (model cited `{', '.join(c.model_cited)}`)" if c.model_cited else ""
        missing = f", not in the source: {', '.join(c.new_terms)}" if c.new_terms else ""
        checked = f", checked as: _{c.checked}_" if c.checked else ""
        st.markdown(f"{mark} {c.text}{fix} (entailment {c.score:.2f}{missing}{checked})")
        citations(c.cited, report, f"{key}-{n}")
    for d in report.declined:
        st.caption(f":material/block: Declined: {d or 'the whole question'}")


def answer(name: str, query: str):
    """The report for this question on this corpus, cached for the browser session."""
    reports = st.session_state.setdefault("reports", {})
    if (name, query) not in reports:
        sear.use_corpus(folders[name])
        reports[name, query] = sear.answer(query)
    return reports[name, query]


sear = get_sear()
folders = corpora()
label = lambda name: "main" if name == "corpus" else name
st.title("SEAR: grounded answers or an honest 'not supported'")

with st.container(horizontal=True, vertical_alignment="bottom"):
    current = sear.chunks_path.parent.name
    name = st.segmented_control("Corpus", list(folders), default=current if current in folders else "corpus",
                                format_func=label, required=True, key="corpus")
    compare = st.toggle("Compare corpora", help="Ask one question on two corpora; shows citations and entailment scores side by side")
    narrator_on = st.toggle("Narrator", value=sear.cfg["narrator"]["enabled"], disabled=compare)
other = None
if compare:
    rest = [n for n in folders if n != name]
    other = st.segmented_control("Compare with", rest, default="poisoned" if "poisoned" in rest else rest[0],
                                 format_func=label, required=True, key="other")
query = st.text_input("Ask the corpus")
narrations = st.session_state.setdefault("narrations", {})

if query and compare and other:
    for col, n in zip(st.columns(2), (name, other)):
        with col:
            st.subheader(label(n))
            r = answer(n, query)
            if r.failed:
                st.info(r.failure_message)
            grounded_report(r, f"cmp-{n}")
            with st.expander("Retrieved chunks"):
                for ch in r.retrieved:
                    st.markdown(f"**{ch.id}**: {ch.text}")
elif query:
    # Cached per corpus and question, so flipping the narrator (a full rerun) never regenerates the report.
    report = answer(name, query)
    if narrator_on and (name, query) not in narrations:
        sear.use_corpus(folders[name])
        narrations[name, query] = sear.narrate(report)
    narration = narrations[name, query] if narrator_on else None

    if report.failed:
        st.info(report.failure_message)
    if narration and (report.failed or narration.skipped):
        st.caption(f"Narrator skipped: {narration.skipped}")
    elif narration:
        st.subheader("Summary")
        for s in narration.sentences:
            cites = f" `{' '.join(s.cited)}`" if s.cited else ""  # no empty code span for uncited sentences
            st.markdown(f"{s.text}{cites}{badge(s)}")
            if s.cited:
                with st.expander("Sources", expanded=False):
                    for h, c, chunks in report.chain(s.cited, narration.sources):
                        st.markdown(f"**{h}** {c.text}")
                        if h in narration.unbacked:
                            st.caption(":orange[No cited chunk or pair entails this sentence on its own: it was verified "
                                       "only as part of its paragraph. These are all the chunks the paragraph cited.]")
                        for ch in chunks:
                            st.markdown(f"`{ch.id}` {ch.text}")
        if narration.unknown_ids:
            st.warning(f"Narrator cited IDs not in the report: {', '.join(narration.unknown_ids)}")
        st.divider()

    if not report.failed:
        st.subheader("Grounded report")
    grounded_report(report, "report")

    with st.expander("Retrieved chunks"):
        for ch in report.retrieved:
            st.markdown(f"**{ch.id}**: {ch.text}")
