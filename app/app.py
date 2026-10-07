import streamlit as st

from sear_demo.config import load_config
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


sear = get_sear()
st.title("SEAR: grounded answers or an honest 'not supported'")

narrator_on = st.toggle("Narrator", value=sear.cfg["narrator"]["enabled"])
query = st.text_input("Ask the corpus")
reports = st.session_state.setdefault("reports", {})
narrations = st.session_state.setdefault("narrations", {})

if query:
    # Cached per query so flipping the toggle (a full rerun) never regenerates the report.
    if query not in reports:
        reports[query] = sear.answer(query)
    report = reports[query]
    if narrator_on and query not in narrations:
        narrations[query] = sear.narrate(report)
    narration = narrations[query] if narrator_on else None

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
        for c in report.claims:
            mark = "✅" if c.verified else "✖️ dropped"
            fix = f" (model cited `{', '.join(c.model_cited)}`)" if c.model_cited else ""
            missing = f", not in the source: {', '.join(c.new_terms)}" if c.new_terms else ""
            checked = f", checked as: _{c.checked}_" if c.checked else ""
            st.markdown(f"{mark} {c.text} `{', '.join(c.cited)}`{fix} (entailment {c.score:.2f}{missing}{checked})")

    with st.expander("Retrieved chunks"):
        for ch in report.retrieved:
            st.markdown(f"**{ch.id}**: {ch.text}")
