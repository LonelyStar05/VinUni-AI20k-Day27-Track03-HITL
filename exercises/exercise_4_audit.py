"""Exercise 4 — Structured SQLite audit trail + durable checkpointer.

Zero setup — SQLite stores everything in a single file (`./hitl_audit.db`).
The audit_events schema is created automatically on first connection.

Goals:

1. Use AsyncSqliteSaver so the graph can resume after a crash.
2. Define and emit an `AuditEntry` (common/schemas.py) for every meaningful step,
   so the full session can be replayed.
3. Verify with `uv run python -m audit.replay --thread <id>`.

This module is also the graph used by the Streamlit UI (`app.py`), so it exposes
`build_graph(checkpointer)` and keeps all node logic free of terminal I/O apart
from progress prints.

Audit rows emitted per branch:

    auto_approve   : fetch_pr → analyze → route → auto_approve
    human_approval : fetch_pr → analyze → route → human_approval(pending)
                     → human_approval(approve|reject|edit) → commit
    escalate       : fetch_pr → analyze → route → escalate(pending)
                     → escalate(escalate) → synthesize → commit
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.checkpoint import open_checkpointer
from common.db import has_event, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.prompts import build_edit_messages, build_review_messages, build_synthesis_messages
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"


def _ms_since(t0: float) -> int:
    return max(0, int((time.monotonic() - t0) * 1000))


def _reviewer_id(resp: dict | None) -> str | None:
    """Who answered the interrupt: explicit id from the UI, else $GITHUB_USER."""
    rid = (resp or {}).get("reviewer_id") or os.environ.get("GITHUB_USER")
    return rid.strip() if isinstance(rid, str) and rid.strip() else None


def _dry_run() -> bool:
    """HITL_DRY_RUN=1 → render the comment but don't POST it to GitHub."""
    return os.environ.get("HITL_DRY_RUN", "").lower() in {"1", "true", "yes"}


async def audit(state, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the `audit_events` table.

    `thread_id` and `pr_url` are taken from `state` so callers only build
    the entry itself.
    """
    await write_audit_event(thread_id=state["thread_id"], pr_url=state["pr_url"], entry=entry)


async def audit_pending_once(state, entry: AuditEntry) -> None:
    """Audit a 'pending' row exactly once per (thread, action).

    A node that calls interrupt() is re-executed from the top on resume, so a
    plain audit() before interrupt() would write the pending row twice.
    """
    if await has_event(thread_id=state["thread_id"], action=entry.action, decision="pending"):
        return
    await audit(state, entry)


async def _wait_ms(state, action: str, fallback_t0: float) -> int:
    """Time the human took to respond = now − timestamp of the pending row."""
    ts = await has_event(thread_id=state["thread_id"], action=action, decision="pending")
    if ts:
        try:
            started = datetime.fromisoformat(ts)
            return max(0, int((datetime.now(timezone.utc) - started).total_seconds() * 1000))
        except ValueError:
            pass
    return _ms_since(fallback_t0)


# ─── Reference example — read this carefully ───────────────────────────────
async def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = await asyncio.to_thread(fetch_pr, state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    # We've only fetched the diff, not analyzed it. So:
    #   - confidence is unknown → 0.0
    #   - risk_level can't be derived from confidence yet → "med" as neutral default
    #   - decision is "pending" — nothing has been decided
    #   - reviewer_id is None — no human is involved at this stage
    #   - reason is a short human-readable summary of what happened
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=_ms_since(t0),
    ))
    return {
        "pr_title": pr.title,
        "pr_author": pr.author,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }
# ───────────────────────────────────────────────────────────────────────────


async def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke(build_review_messages(state))
    console.print(f"  [green]✓[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="analyze",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",                      # routing hasn't happened yet
        reason=a.confidence_reasoning,
        execution_time_ms=_ms_since(t0),
    ))
    return {"analysis": a}


async def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    # First row where `decision` is real. Map the branch onto the AuditEntry vocabulary.
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="route",
        confidence=c,
        risk_level=risk_level_for(c),
        decision={"auto_approve": "auto", "human_approval": "pending", "escalate": "escalate"}[decision],
        reason=(
            f"branch={decision} (auto ≥ {AUTO_APPROVE_THRESHOLD:.0%}, "
            f"escalate < {ESCALATE_THRESHOLD:.0%})"
        ),
        execution_time_ms=_ms_since(t0),
    ))
    return {"decision": decision}


async def node_human_approval(state):
    t0 = time.monotonic()
    a = state["analysis"]

    # #1 — BEFORE the interrupt: request sent, no human has responded yet.
    # (Idempotent: this line re-runs on resume.)
    await audit_pending_once(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason="Approval requested from reviewer",
        execution_time_ms=_ms_since(t0),
    ))

    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "pr_title": state.get("pr_title"),
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
        # Present only when this is the final confirm of a refined (post-escalation) review.
        "refined": bool(state.get("escalation_answers")),
        "initial_confidence": state.get("initial_confidence"),
        "escalation_answers": state.get("escalation_answers"),
    }) or {}

    choice = resp.get("choice")
    if choice not in {"approve", "reject", "edit"}:
        raise ValueError(f"Invalid reviewer choice: {choice!r}")
    feedback = (resp.get("feedback") or "").strip() or None

    # #2 — AFTER resume: we know who answered, what they chose, and why.
    # execution_time_ms here = how long the human took to respond.
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer_id(resp),
        decision=choice,
        reason=feedback or f"Reviewer chose {choice}",
        execution_time_ms=await _wait_ms(state, "human_approval", t0),
    ))
    return {"human_choice": choice, "human_feedback": feedback}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    if a.risk_factors:
        lines.append("**Risk factors:** " + "; ".join(a.risk_factors))
        lines.append("")
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    lines.append(f"\n<sub>{AGENT_ID} · thread `{state.get('thread_id', '?')}`</sub>")
    return "\n".join(lines)


async def _post(state) -> tuple[str, str | None, str]:
    """Post the review. Returns (final_action, comment_url, body)."""
    body = _render_comment_body(state)
    if _dry_run():
        console.print("  [yellow]·[/yellow] HITL_DRY_RUN=1 — comment rendered but not posted")
        return "committed", None, body
    try:
        url = await asyncio.to_thread(post_review_comment, state["pr_url"], body)
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return "committed", url, body
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed", None, body


async def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    t0 = time.monotonic()
    a: PRAnalysis = state["analysis"]
    choice = state.get("human_choice")
    updates: dict = {}

    # Paths that converge here:
    #   1. human_approval(approve)        → post as-is
    #   2. human_approval(edit, feedback) → LLM rewrites per feedback (bonus 4), then post
    #   3. escalate → synthesize          → post the refined review
    #   4. escalate → synthesize → human_approval (confirm_refined=True, used by the UI)
    #                                     → follow the reviewer's final choice
    #   human_approval(reject)            → nothing posted
    if choice == "edit" and state.get("human_feedback"):
        llm = get_llm().with_structured_output(PRAnalysis)
        with console.status("[dim]LLM applying reviewer edits...[/dim]"):
            a = await llm.ainvoke(build_edit_messages(state, a, state["human_feedback"]))
        updates["analysis"] = a
        state = {**state, "analysis": a}

    if choice != "reject" and (state.get("escalation_answers") or choice in {"approve", "edit"}):
        action, url, body = await _post(state)
        if choice == "edit" and action == "committed":
            action = "committed_with_edit"
        elif state.get("escalation_answers") and action == "committed":
            action = "committed_after_escalation"
        decision = choice or "escalate"
        updates.update({"posted_comment_body": body, "comment_url": url})
    else:
        console.print(f"  [yellow]·[/yellow] skipping comment (choice={choice})")
        action, url, decision = "rejected", None, "reject"

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="commit",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision=decision,
        reason=(
            f"{action}" + (f" → {url}" if url else "") + (" (dry-run)" if _dry_run() and action != "rejected" else "")
            + f"; {len(a.comments)} comment(s)"
        ),
        execution_time_ms=_ms_since(t0),
    ))
    return {"final_action": action, **updates}


async def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    t0 = time.monotonic()
    a = state["analysis"]
    action, url, body = await _post(state)
    # No human involved → decision "auto", reviewer_id None.
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_approve",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="auto",
        reason=f"{action}" + (f" → {url}" if url else "") + f"; {a.confidence_reasoning}",
        execution_time_ms=_ms_since(t0),
    ))
    return {"final_action": f"auto_{action}", "posted_comment_body": body, "comment_url": url}


async def node_escalate(state):
    t0 = time.monotonic()
    a = state["analysis"]
    questions = a.escalation_questions or [
        "What is the intent of this PR, and which parts are security-sensitive?",
        "Are there migration / backwards-compatibility concerns the diff doesn't show?",
    ]

    # #1 — BEFORE the interrupt (reviewer hasn't answered yet). Idempotent on resume.
    await audit_pending_once(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=f"Escalated with {len(questions)} question(s): " + " | ".join(questions),
        execution_time_ms=_ms_since(t0),
    ))

    resp = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "pr_title": state.get("pr_title"),
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "comments": [c.model_dump() for c in a.comments],
        "questions": questions,
        "diff_preview": state["pr_diff"][:2000],
    }) or {}

    # Accept either {question: answer} or {"answers": {...}, "reviewer_id": "..."}.
    raw = resp.get("answers", resp) if isinstance(resp, dict) else {}
    answers = {q: str(raw.get(q, "")).strip() for q in questions}

    # #2 — AFTER resume: we have the answers.
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer_id(resp),
        decision="escalate",
        reason=" | ".join(f"Q: {q} A: {ans or '-'}" for q, ans in answers.items()),
        execution_time_ms=await _wait_ms(state, "escalate", t0),
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state):
    console.print("[cyan]→ synthesize[/cyan]")
    t0 = time.monotonic()
    initial: PRAnalysis = state["analysis"]
    answers = state.get("escalation_answers") or {}
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke(build_synthesis_messages(state, initial, answers))
    console.print(
        f"  [green]✓[/green] refined confidence={refined.confidence:.0%} (was {initial.confidence:.0%})"
    )
    # Use the NEW confidence — normally higher than the initial analysis.
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="synthesize",
        confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence),
        decision="pending",                     # commit decides next
        reason=f"confidence {initial.confidence:.2f} → {refined.confidence:.2f}; {refined.confidence_reasoning}",
        execution_time_ms=_ms_since(t0),
    ))
    # node_commit will run next and post the refined review to the PR.
    return {"analysis": refined, "initial_confidence": initial.confidence}


def build_graph(checkpointer, confirm_refined: bool = False):
    """Compile the full HITL graph.

    confirm_refined=False (CLI default, as in the README diagram):
        escalate → synthesize → commit
    confirm_refined=True (Streamlit UI): the refined review goes back through
        human_approval for a final Approve / Reject / Edit before posting.
    """
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "human_approval" if confirm_refined else "commit")
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            f"[bold]Confidence:[/bold] {payload['confidence']:.0%}\n"
            f"[dim]{payload['confidence_reasoning']}[/dim]\n\n{payload['summary']}",
            title="Approval request", border_style="green",
        ))
        for c in payload.get("comments", []):
            console.print(f"  [{c['severity']}] {c['file']}:{c.get('line') or '?'} — {c['body']}")
        choice = ""
        while choice not in {"approve", "reject", "edit"}:
            choice = console.input("approve/reject/edit? ").strip().lower()
        feedback = console.input("Feedback: ").strip() if choice != "approve" else ""
        return {"choice": choice, "feedback": feedback}
    console.print(Panel.fit(
        f"[bold]Confidence:[/bold] {payload['confidence']:.0%}\n"
        f"[dim]{payload['confidence_reasoning']}[/dim]\n\n{payload['summary']}",
        title="Escalation — reviewer input needed", border_style="yellow",
    ))
    for r in payload.get("risk_factors", []):
        console.print(f"  [red]•[/red] {r}")
    return {q: console.input(f"\nQ: {q}\nA: ").strip() for q in payload["questions"]}


async def run(pr_url: str, thread_id: str | None):
    resuming = thread_id is not None
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with open_checkpointer() as cp:
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        snapshot = await app.aget_state(cfg) if resuming else None
        if snapshot is not None and snapshot.interrupts:
            # Resume a session that was paused (e.g. process killed while waiting for a human).
            console.print("[dim]Resuming paused session from checkpoint...[/dim]")
            result = {"__interrupt__": snapshot.interrupts}
        elif snapshot is not None and snapshot.values and not snapshot.next:
            console.print("[dim]Thread already finished.[/dim]")
            result = snapshot.values
        else:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume an existing thread")
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
