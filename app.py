"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Wraps the LangGraph from exercise 4 (same nodes, same SQLite checkpointer and
audit trail) in a web UI that adapts to the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    ≥ 73%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer, then the
                                  refined review for a final confirm

Session persistence: the active `thread_id` lives in `st.session_state` *and*
in the URL (`?thread=<id>`), so a full page reload — or opening the link later,
even after restarting Streamlit — restores the session from the checkpointer.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from audit.replay import fetch_recent_threads
from common.checkpoint import open_checkpointer
from common.db import replay_events
from common.schemas import AUTO_APPROVE_THRESHOLD, ESCALATE_THRESHOLD
from exercises.exercise_4_audit import build_graph


load_dotenv()

SEVERITY_ICON = {"nit": "⚪", "suggestion": "🔵", "issue": "🟠", "blocker": "🔴"}
RISK_ICON = {"low": "🟢", "med": "🟡", "high": "🔴", "-": "⚪"}


# ─── Session state ─────────────────────────────────────────────────────────
for key, default in {
    "thread_id": None,
    "pr_url": "",
    "interrupt_payload": None,
    "final": None,
    "error": None,
}.items():
    st.session_state.setdefault(key, default)


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", page_icon="🔎", layout="wide")
st.title("🔎 HITL PR Review Agent")
st.caption(
    f"Auto-approve ≥ {AUTO_APPROVE_THRESHOLD:.0%} · Human approval "
    f"{ESCALATE_THRESHOLD:.0%}–{AUTO_APPROVE_THRESHOLD:.0%} · Escalate < {ESCALATE_THRESHOLD:.0%}"
)


# ─── Graph driver ──────────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None) -> dict:
    """Invoke the graph once. Returns the final state or {'__interrupt__': ...}."""
    async with open_checkpointer() as cp:
        app = build_graph(cp, confirm_refined=True)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


async def load_thread(thread_id: str) -> tuple[dict, dict | None, dict | None]:
    """Restore a session from the checkpointer: (values, interrupt_payload, final)."""
    async with open_checkpointer() as cp:
        app = build_graph(cp, confirm_refined=True)
        snap = await app.aget_state({"configurable": {"thread_id": thread_id}})
    values = dict(snap.values or {})
    if snap.interrupts:
        return values, snap.interrupts[0].value, None
    if values and not snap.next:
        return values, None, values
    return values, None, None


def apply_result(result: dict) -> None:
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result


def open_thread(thread_id: str) -> None:
    """Make `thread_id` the active session (from sidebar click or ?thread=...)."""
    values, payload, final = asyncio.run(load_thread(thread_id))
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = values.get("pr_url", "")
    st.session_state.interrupt_payload = payload
    st.session_state.final = final
    st.session_state.error = None
    st.query_params["thread"] = thread_id


# Restore from the URL on a fresh browser session (page reload / shared link).
_qp_thread = st.query_params.get("thread")
if _qp_thread and st.session_state.thread_id != _qp_thread:
    try:
        open_thread(_qp_thread)
    except Exception as e:  # noqa: BLE001
        st.session_state.error = f"Could not restore thread {_qp_thread}: {e}"


# ─── Sidebar — reviewer identity + recent sessions ─────────────────────────
with st.sidebar:
    st.header("Reviewer")
    reviewer_id = st.text_input(
        "GitHub username", value=os.environ.get("GITHUB_USER", ""),
        help="Written to audit_events.reviewer_id on every HITL decision.",
    )
    dry_run = st.toggle(
        "Dry run (don't post to GitHub)",
        value=os.environ.get("HITL_DRY_RUN", "").lower() in {"1", "true", "yes"},
    )
    os.environ["HITL_DRY_RUN"] = "1" if dry_run else "0"

    st.divider()
    st.header("Recent sessions")
    try:
        threads = asyncio.run(fetch_recent_threads(limit=15))
    except Exception as e:  # noqa: BLE001
        threads = []
        st.caption(f"(audit DB unavailable: {e})")
    if not threads:
        st.caption("No sessions yet — run a review.")
    for t in threads:
        pr_short = t["pr_url"].split("github.com/")[-1].replace("/pull/", " #")
        active = t["thread_id"] == st.session_state.thread_id
        label = (
            f"{RISK_ICON.get(t['worst_risk'], '⚪')} {pr_short}\n\n"
            f"{t['last_action']} · {t['last_decision']} · {t['events']} events"
        )
        if st.button(label, key=f"thread-{t['thread_id']}", width="stretch",
                     type="primary" if active else "secondary",
                     help=f"thread {t['thread_id']}\nlast event {t['last_event']}"):
            open_thread(t["thread_id"])
            st.rerun()
    if st.session_state.thread_id and st.button("＋ New review", width="stretch"):
        for k in ("thread_id", "interrupt_payload", "final", "error"):
            st.session_state[k] = None
        st.session_state.pr_url = ""
        st.query_params.clear()
        st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", type="primary")


# ─── Shared bits ───────────────────────────────────────────────────────────
def render_comments(comments: list[dict]) -> None:
    if not comments:
        st.caption("No line comments.")
    for c in comments:
        icon = SEVERITY_ICON.get(c["severity"], "•")
        st.markdown(f"{icon} **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")


def render_header(payload: dict) -> None:
    if payload.get("pr_title"):
        st.markdown(f"**{payload['pr_title']}** · [{payload['pr_url']}]({payload['pr_url']})")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket (or final confirm after escalation): review + 3 buttons."""
    conf = payload["confidence"]
    tid = st.session_state.thread_id
    with st.container(border=True):
        if payload.get("refined"):
            init = payload.get("initial_confidence")
            st.subheader("Refined review — final confirm")
            st.caption(
                "The agent re-analysed the PR with your answers"
                + (f" · confidence {init:.0%} → **{conf:.0%}**" if init is not None else f" · confidence {conf:.0%}")
            )
        else:
            st.subheader(f"🟡 Approval requested — confidence {conf:.0%}")
        render_header(payload)
        st.progress(conf, text=f"LLM confidence {conf:.0%}")
        st.info(payload["confidence_reasoning"], icon="🧠")
        st.markdown(payload["summary"])
        if payload.get("risk_factors"):
            st.warning("**Risk factors:** " + " · ".join(payload["risk_factors"]))

        st.markdown("##### Proposed review comments")
        render_comments(payload.get("comments", []))

        if payload.get("escalation_answers"):
            with st.expander("Your answers to the escalation questions"):
                for q, a in payload["escalation_answers"].items():
                    st.markdown(f"**Q:** {q}\n\n**A:** {a or '—'}")

        with st.expander("Diff preview", expanded=not payload.get("refined")):
            st.code(payload.get("diff_preview", ""), language="diff")

        feedback = st.text_area(
            "Feedback / edit instructions",
            key=f"approval_feedback-{tid}-{payload.get('refined')}",
            placeholder="Optional for Approve/Reject. Required for Edit — e.g. "
                        "“downgrade the migration comment to a suggestion and mention ALTER TABLE”.",
        )
        col1, col2, col3 = st.columns(3)
        base = {"feedback": feedback.strip(), "reviewer_id": reviewer_id.strip() or None}
        if col1.button("✅ Approve & post", type="primary", width="stretch"):
            return {"choice": "approve", **base}
        if col2.button("❌ Reject", width="stretch"):
            return {"choice": "reject", **base}
        if col3.button("✏️ Edit & post", width="stretch"):
            if not feedback.strip():
                st.error("Write your edit instructions in the feedback box first.")
                return None
            return {"choice": "edit", **base}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: risk factors + question form. Returns answers or None."""
    conf = payload["confidence"]
    tid = st.session_state.thread_id
    with st.container(border=True):
        st.subheader(f"🔴 Strong escalation — confidence {conf:.0%}")
        render_header(payload)
        st.progress(conf, text=f"LLM confidence {conf:.0%}")
        st.info(payload["confidence_reasoning"], icon="🧠")
        if payload.get("risk_factors"):
            st.error("**Risk factors**\n\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
        st.markdown(payload["summary"])
        if payload.get("comments"):
            with st.expander("Initial (unconfirmed) comments"):
                render_comments(payload["comments"])
        with st.expander("Diff preview"):
            st.code(payload.get("diff_preview", ""), language="diff")

        st.markdown("##### The agent needs your input")
        with st.form(f"escalation-{tid}"):
            answers: dict[str, str] = {}
            for i, q in enumerate(payload["questions"]):
                answers[q] = st.text_area(f"**Q{i + 1}.** {q}", key=f"esc-{tid}-{i}")
            if st.form_submit_button("Submit answers → re-analyse", type="primary"):
                if not any(a.strip() for a in answers.values()):
                    st.error("Answer at least one question.")
                    return None
                return {
                    "answers": {q: a.strip() for q, a in answers.items()},
                    "reviewer_id": reviewer_id.strip() or None,
                }
    return None


def render_final(final: dict) -> None:
    action = final.get("final_action") or "?"
    analysis = final.get("analysis")
    url = final.get("comment_url")
    dry = os.environ.get("HITL_DRY_RUN") == "1"
    with st.container(border=True):
        if action.startswith("auto") and "failed" not in action:
            st.subheader("🟢 Auto-approved — no human needed")
            st.success(f"{action}" + (" (dry run — nothing posted)" if dry and not url else ""))
        elif action.startswith("committed"):
            st.subheader("✅ Review posted")
            st.success(action + (" (dry run — nothing posted)" if dry and not url else ""))
        elif action == "rejected":
            st.subheader("❌ Rejected")
            st.warning("Rejected — no comment posted.")
        elif "failed" in action:
            st.subheader("⚠️ Posting failed")
            st.error(f"{action} — check GITHUB_TOKEN scope (`public_repo`).")
        else:
            st.info(f"final_action = {action}")

        if analysis is not None:
            conf = analysis.confidence
            if final.get("initial_confidence") is not None:
                st.caption(f"Confidence {final['initial_confidence']:.0%} → {conf:.0%} after reviewer Q&A")
            st.progress(conf, text=f"Final confidence {conf:.0%}")
            st.markdown(analysis.summary)
        if url:
            st.link_button("View comment on GitHub ↗", url)
        if final.get("posted_comment_body"):
            with st.expander("Comment body"):
                st.markdown(final["posted_comment_body"])


def render_audit_timeline(thread_id: str) -> None:
    events = asyncio.run(replay_events(thread_id))
    if not events:
        return
    with st.expander(f"Audit trail · {len(events)} events", expanded=False):
        st.dataframe(
            [
                {
                    "time": e["timestamp"][11:19],
                    "action": e["action"],
                    "decision": e["decision"],
                    "confidence": round(e["confidence"], 2),
                    "risk": f"{RISK_ICON.get(e['risk_level'], '')} {e['risk_level']}",
                    "reviewer": e["reviewer_id"] or "—",
                    "ms": e["execution_time_ms"],
                    "reason": e["reason"],
                }
                for e in events
            ],
            hide_index=True, width="stretch",
        )
        st.caption(f"`uv run python -m audit.replay --thread {thread_id}`")


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url.strip()
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.error = None
    st.query_params["thread"] = st.session_state.thread_id
    try:
        with st.spinner("Fetching PR + asking the LLM..."):
            apply_result(asyncio.run(run_graph(st.session_state.pr_url, st.session_state.thread_id)))
    except Exception as e:  # noqa: BLE001
        st.session_state.error = f"{type(e).__name__}: {e}"
    st.rerun()

if st.session_state.error:
    st.error(st.session_state.error)

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        try:
            spinner = "Re-analysing with your answers..." if kind == "escalation" else "Resuming..."
            with st.spinner(spinner):
                apply_result(asyncio.run(run_graph(
                    st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
                )))
            st.session_state.error = None
        except Exception as e:  # noqa: BLE001
            st.session_state.error = f"{type(e).__name__}: {e}"
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    render_final(st.session_state.final)

if st.session_state.thread_id:
    render_audit_timeline(st.session_state.thread_id)
    st.caption(f"thread_id = `{st.session_state.thread_id}`")
