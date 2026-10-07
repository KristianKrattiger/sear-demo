import streamlit as st

from sear_demo.config import load_config
from sear_demo.pipeline import Sear

st.set_page_config(page_title="SEAR demo", layout="wide")


@st.cache_resource
def get_sear():
    return Sear(load_config())


sear = get_sear()
st.title("SEAR: grounded answers or an honest 'not supported'")

narrator_on = st.toggle("Narrator", value=sear.cfg["narrator"]["enabled"])
query = st.text_input("Ask the corpus")

if query:
    result = sear.ask(query, narrator=narrator_on)
    report = result.report

    if report.failed:
        st.info(report.failure_message)
    else:
        if result.narration:
            st.subheader("Summary")
            for s in result.narration.sentences:
                badge = "" if s.verified else "  ⚠️ not supported by the grounded report"
                st.markdown(f"{s.text} `{', '.join(s.cited)}`{badge}")
            if result.narration.unknown_ids:
                st.warning(f"Narrator cited IDs not in the report: {', '.join(result.narration.unknown_ids)}")
            st.divider()

        st.subheader("Grounded report")
        for c in report.claims:
            mark = "✅" if c.verified else "✖️ dropped"
            fix = f" (model cited `{', '.join(c.model_cited)}`)" if c.model_cited else ""
            st.markdown(f"{mark} {c.text} `{', '.join(c.cited)}`{fix} (entailment {c.score:.2f})")

    with st.expander("Retrieved chunks"):
        for ch in report.retrieved:
            st.markdown(f"**{ch.id}**: {ch.text}")
