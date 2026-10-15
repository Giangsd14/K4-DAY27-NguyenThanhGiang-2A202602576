"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread" in st.query_params and st.query_params["thread"]:
    if "thread_id" not in st.session_state or st.session_state.thread_id != st.query_params["thread"]:
        st.session_state.thread_id = st.query_params["thread"]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review Agent", page_icon="🤖", layout="wide")
st.title("🤖 Human-in-the-Loop PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
async def get_recent_sessions():
    try:
        async with db_conn() as conn:
            async with conn.execute(
                """
                SELECT thread_id,
                       pr_url,
                       MIN(timestamp)        AS started,
                       MAX(timestamp)        AS last_event,
                       MAX(risk_level)       AS worst_risk,
                       COUNT(*)              AS events
                  FROM audit_events
                 GROUP BY thread_id, pr_url
                 ORDER BY MAX(timestamp) DESC
                 LIMIT 15
                """
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


with st.sidebar:
    st.header("📋 Recent Sessions")
    sessions = asyncio.run(get_recent_sessions())
    if sessions:
        for s in sessions:
            risk_color = "🟢" if s["worst_risk"] == "low" else ("🟡" if s["worst_risk"] == "med" else "🔴")
            pr_label = s["pr_url"].split("/")[-2] + "/" + s["pr_url"].split("/")[-1] if "/" in s["pr_url"] else s["pr_url"]
            btn_label = f"{risk_color} {pr_label} ({s['thread_id'][:6]})"
            if st.button(btn_label, key=f"session_btn_{s['thread_id']}", use_container_width=True):
                st.session_state.thread_id = s["thread_id"]
                st.session_state.pr_url = s["pr_url"]
                st.query_params["thread"] = s["thread_id"]
                st.rerun()
            st.caption(f"Last: {s.get('last_event', '')[:19]} | Events: {s['events']}")
            st.divider()
    else:
        st.caption("No review sessions recorded yet.")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "Pull Request URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("🚀 Run Review", type="primary")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.info(f"### ⚖️ Human Approval Requested (Confidence: {conf:.0%})")
    st.markdown(f"**Reasoning:** _{payload.get('confidence_reasoning', '')}_")
    st.markdown(f"**Summary:** {payload.get('summary', '')}")

    comments = payload.get("comments", [])
    if comments:
        st.markdown("#### 💬 Proposed Review Comments:")
        for c in comments:
            severity_color = "red" if c.get("severity") in ("blocker", "issue") else "blue"
            st.markdown(f"- :{severity_color}[**[{c.get('severity', 'note').upper()}]**] `{c.get('file', '')}:{c.get('line') or '?'}` — {c.get('body', '')}")

    if payload.get("diff_preview"):
        with st.expander("🔍 View Diff Preview"):
            st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Reviewer Notes / Feedback (optional)", key="approval_feedback")

    col1, col2, col3 = st.columns(3)
    if col1.button("✅ Approve & Post", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("❌ Reject Review", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("✏️ Edit Review", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.warning(f"### ⚠️ Strong Escalation Required (Confidence: {conf:.0%})")
    st.markdown(f"**Reasoning:** _{payload.get('confidence_reasoning', '')}_")
    
    if payload.get("risk_factors"):
        st.error("🚨 **Identified Risks:**\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
    
    st.markdown(f"**Summary:** {payload.get('summary', '')}")
    st.markdown("#### ❓ The Agent requires clarification on the following questions:")

    with st.form("escalation"):
        answers: dict[str, str] = {}
        for i, q in enumerate(payload.get("questions", [])):
            answers[q] = st.text_area(f"Q{i+1}: {q}", key=f"esc_q_{i}")
        submitted_answers = st.form_submit_button("📤 Submit Answers & Re-synthesize", type="primary")
        if submitted_answers:
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR & analyzing diff with AI Agent..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result


# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming agent review..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()


# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"🎉 **Review Successfully Posted!** (Action: `{action}`)")
        st.markdown(f"🔗 [View Pull Request on GitHub]({st.session_state.pr_url})")
    elif action == "rejected":
        st.warning(f"🛑 **Review Rejected** — No comment was posted to GitHub.")
    else:
        st.info(f"ℹ️ Final Status: `{action}`")

    if "analysis" in final:
        st.markdown(f"**Final Confidence:** `{final['analysis'].confidence:.0%}`")

    st.caption(f"Session Thread ID: `{st.session_state.thread_id}`  •  Audit replay: `uv run python -m audit.replay --thread {st.session_state.thread_id}`")
