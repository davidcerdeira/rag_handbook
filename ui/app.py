"""Phase 5: Streamlit UI for the HR handbook RAG API.

Run (with the API already running in another terminal):
    streamlit run ui/app.py
"""
import os
import re

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
CITATION = re.compile(r"\[(\d+)\]")

EXAMPLES = {
    "🇮🇪 Paternity leave in Ireland": "How much paternity leave do I get in Ireland?",
    "👶 Time off for a new baby": "Can I take time off when my baby is born?",
    "❓ What is PRSI?": "What is PRSI?",
    "🍕 Pizza in Lisbon": "What's the best pizza place in Lisbon?",
}

BADGES = {
    "High": ("🟢", "High confidence"),
    "Medium": ("🟡", "Medium confidence"),
    "Low": ("🔴", "Low confidence"),
    "None": ("⚪", "Not covered by the handbook"),
}


def ask_api(question: str) -> dict:
    response = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60)
    if response.status_code == 422:
        raise ValueError("Please enter a question between 3 and 500 characters.")
    response.raise_for_status()
    return response.json()


def md_safe(text: str) -> str:
    """Streamlit reads $...$ as maths formulas; escape it so '$500' displays normally."""
    return text.replace("$", r"\$")


def use_example(question: str) -> None:
    st.session_state.question = question
    st.session_state.run = True

def render_trace(data: dict) -> None:
    """What the pipeline actually did, stage by stage, with real timings."""
    t = data["timings"]
    sources = data["sources"]
    best = sources[0]["score"] if sources else 0.0
    st.markdown(f"✓ **Searched the handbook:** {len(sources)} sections found, "
                f"best match {best:.2f} · {t.get('search', 0):.1f}s")
    if data["gated"]:
        st.markdown(f"✗ **Relevance gate:** best match {best:.2f} is below "
                    f"{data['gate_threshold']}, so Claude was not called")
        return
    st.markdown(f"✓ **Relevance gate passed** (threshold {data['gate_threshold']})")
    st.markdown(f"✓ **Claude wrote the answer:** {data['input_tokens']:,} in / "
                f"{data['output_tokens']:,} out tokens · {t.get('answer', 0):.1f}s")
    check = data["citations"]
    if check:
        mark = "✓" if check["supported"] == check["total"] else "⚠"
        st.markdown(f"{mark} **Citations checked:** {check['supported']}/{check['total']} "
                    f"supported · {t.get('check', 0):.1f}s")
    else:
        st.markdown("– **Citation check skipped:** the answer has no citations")
    st.markdown(f"✓ **Confidence:** {data['confidence']['label']}")

def render(data: dict) -> None:
    with st.status(f"Done in {data['latency_s']}s", state="complete", expanded=False):
        render_trace(data)
    conf = data["confidence"]
    icon, label = BADGES.get(conf["label"], ("⚪", conf["label"]))
    st.markdown(f"### {icon} {label}")
    if conf["reasons"]:
        st.caption(" · ".join(conf["reasons"]))

    if data["gated"]:
        st.info(data["answer"])
    else:
        st.markdown(md_safe(data["answer"]))
        if data["truncated"]:
            st.warning("This answer was cut off (it hit the length limit).")

    check = data["citations"]
    if check:
        all_ok = check["supported"] == check["total"]
        show = st.success if all_ok else st.warning
        show(f"{'✅' if all_ok else '⚠️'} {check['supported']}/{check['total']} "
             "claims verified against their cited sources")

    cited = {int(n) for n in CITATION.findall(data["answer"])}
    st.subheader("Closest sections found" if data["gated"] else "Sources")
    for s in data["sources"]:
        tag = "cited" if s["n"] in cited else "retrieved, not cited"
        heading = f" › {s['heading']}" if s["heading"] else ""
        with st.expander(f"[{s['n']}] {s['title']}{heading} · relevance {s['score']:.2f} · {tag}"):
            st.markdown(md_safe(s["text"]))
            st.caption(f"Source: {s['source']}")

    st.caption(f"⏱ {data['latency_s']}s · {data['input_tokens']} in / "
               f"{data['output_tokens']} out tokens")


# ---------- Page ----------

st.set_page_config(page_title="Ask the GitLab HR Handbook", page_icon="📘", layout="centered")
st.title("📘 Ask the GitLab HR Handbook")
st.caption("Hybrid search + reranking + Claude, with every citation checked by a second model.")

with st.sidebar:
    st.header("About")
    st.markdown(
        "Answers questions about GitLab's public HR handbook "
        "(people policies + total rewards).\n\n"
        "- **Search:** meaning (Voyage embeddings) + exact words (BM25), fused with RRF\n"
        "- **Rerank:** cross-encoder picks the best 5 of 20\n"
        "- **Gate:** says *not covered* without calling the LLM when nothing is relevant\n"
        "- **Answer:** Claude, citing every fact\n"
        "- **Check:** a second model verifies each citation"
    )
    try:
        online = requests.get(f"{API_URL}/health", timeout=3).ok
    except requests.RequestException:
        online = False
    st.markdown(f"**API:** {'🟢 online' if online else '🔴 offline'}")

st.write("Try an example:")
for row in (list(EXAMPLES.items())[:2], list(EXAMPLES.items())[2:]):
    for col, (label, q) in zip(st.columns(2), row):
        col.button(label, on_click=use_example, args=(q,), use_container_width=True)

with st.form("ask"):
    st.text_input("Your question", key="question",
                  placeholder="e.g. How many vacation days do I get in the Netherlands?")
    submitted = st.form_submit_button("Ask", type="primary")

if submitted or st.session_state.pop("run", False):
    question = st.session_state.get("question", "").strip()
    st.session_state.pop("result", None)
    with st.spinner("Searching the handbook, writing the answer, checking every citation…"):
        try:
            st.session_state.result = ask_api(question)
        except ValueError as e:
            st.error(str(e))
        except requests.ConnectionError:
            st.error("Can't reach the API. Is `uvicorn src.api:app` running?")
        except requests.RequestException:
            st.error("Something went wrong. Please try again.")

if "result" in st.session_state:
    render(st.session_state.result)