"""Streamlit UI: streamlit run app.py"""
import streamlit as st

from contribrag.agent import ContribAgent
from contribrag.pipeline import index_repo

st.set_page_config(page_title="ContribRAG", page_icon="🧭", layout="wide")
st.title("🧭 ContribRAG — ask a repo, then contribute to it")

if "agent" not in st.session_state:
    st.session_state.agent = ContribAgent()
agent = st.session_state.agent

with st.sidebar:
    st.header("1 · Index a repo")
    source = st.text_input("GitHub URL or local path", placeholder="https://github.com/owner/repo")
    if st.button("Ingest") and source:
        with st.spinner("Cloning, chunking, indexing…"):
            st.session_state.repo_label = index_repo(source)["repo"]
        st.success(f"Indexed {st.session_state.get('repo_label', source)}")
    st.header("2 · Your skills")
    skills = st.text_input("Skills / interests", value="python, llm, rag, agents")

repo = st.session_state.get("repo_label")
tab_ask, tab_match = st.tabs(["Ask the repo", "Contribute mode"])

with tab_ask:
    q = st.text_input("Question", placeholder="Where is authentication handled?")
    if st.button("Ask") and q:
        out = agent.run(question=q, repo=repo)
        st.markdown(out["answer"])
        st.caption(f"mode: {out['mode']} · citations verified: {out['verification']['ok']}")
        with st.expander("Sources"):
            for c in out["chunks"]:
                st.write(f"[{c.citation}] {c.symbol}")

with tab_match:
    src = st.text_input("Repo URL for issue matching", value=source or "")
    if st.button("Match me to issues") and src:
        out = agent.run(question="which issue should I contribute to",
                        source=src, skills=skills)
        for m in out.get("matches", []):
            st.markdown(f"**#{m['number']} {m['title']}** — score {m['score']}")
            st.write(f"Why: {m['why']}  ·  {m['url']}")
