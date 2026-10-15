"""End-to-end tests for exercises 1–4, fully offline (fake LLM + fake GitHub).

    uv run pytest -q
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid

import pytest
from common.checkpoint import open_checkpointer
from langgraph.types import Command

from tests.fakes import PR2_QUESTIONS, FakeGitHub, patch_module

PR1 = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"
PR2 = "https://github.com/VinUni-AI20k/PR-Demo/pull/2"
PR3 = "https://github.com/VinUni-AI20k/PR-Demo/pull/3"


@pytest.fixture()
def gh():
    return FakeGitHub()


@pytest.fixture()
def db(tmp_path, monkeypatch):
    path = tmp_path / "hitl_audit.db"
    monkeypatch.setenv("HITL_DB_PATH", str(path))
    monkeypatch.setenv("GITHUB_USER", "LonelyStar05")
    monkeypatch.delenv("HITL_DRY_RUN", raising=False)
    return path


def rows(db, thread_id):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    out = [dict(r) for r in con.execute(
        "SELECT * FROM audit_events WHERE thread_id = ? ORDER BY id", (thread_id,))]
    con.close()
    return out


# ─── Exercise 1 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("url,branch", [
    (PR1, "human_approval"), (PR2, "escalate"), (PR3, "auto_approve"),
])
def test_ex1_routes_three_branches(gh, url, branch):
    from exercises import exercise_1_confidence as ex1
    patch_module(ex1, gh)
    final = ex1.build_graph().invoke({"pr_url": url})
    assert final["decision"] == branch


@pytest.mark.parametrize("c,expected", [
    (0.90, "auto_approve"), (0.73, "auto_approve"), (0.72, "human_approval"),
    (0.58, "human_approval"), (0.5799, "escalate"), (0.0, "escalate"),
])
def test_ex1_threshold_edges(c, expected):
    from exercises.exercise_1_confidence import decide
    assert decide(c) == expected


# ─── Exercise 2 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("choice,posted,final", [
    ("approve", 1, "committed"), ("reject", 0, "rejected"), ("edit", 1, "committed_with_edit"),
])
def test_ex2_interrupt_and_resume(gh, choice, posted, final):
    from exercises import exercise_2_hitl as ex2
    patch_module(ex2, gh)
    app = ex2.build_graph()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = app.invoke({"pr_url": PR1, "thread_id": "t"}, cfg)
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "approval_request"
    assert 0.58 <= payload["confidence"] < 0.73
    assert payload["diff_preview"] and payload["comments"]
    result = app.invoke(Command(resume={"choice": choice, "feedback": "note"}), cfg)
    assert "__interrupt__" not in result
    assert result["final_action"] == final
    assert len(gh.posted) == posted


# ─── Exercise 3 ────────────────────────────────────────────────────────────
def test_ex3_escalation_qa_then_refined_commit(gh):
    from exercises import exercise_3_escalation as ex3
    patch_module(ex3, gh)
    app = ex3.build_graph()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = app.invoke({"pr_url": PR2, "thread_id": "t"}, cfg)
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "escalation"
    assert payload["questions"] == PR2_QUESTIONS
    assert payload["risk_factors"]
    answers = {q: f"answer {i}" for i, q in enumerate(payload["questions"])}
    result = app.invoke(Command(resume=answers), cfg)
    assert result["final_action"] == "committed_after_escalation"
    assert result["analysis"].confidence > payload["confidence"]
    assert len(gh.posted) == 1 and "answer 0" in gh.posted[0][1]


# ─── Exercise 4 ────────────────────────────────────────────────────────────
async def _run_ex4(url, answer_fn):
    from exercises import exercise_4_audit as ex4
    tid = str(uuid.uuid4())
    async with open_checkpointer() as cp:
        app = ex4.build_graph(cp)
        cfg = {"configurable": {"thread_id": tid}}
        result = await app.ainvoke({"pr_url": url, "thread_id": tid}, cfg)
        while "__interrupt__" in result:
            result = await app.ainvoke(Command(resume=answer_fn(result["__interrupt__"][0].value)), cfg)
    return tid, result


def test_ex4_auto_branch_audit(gh, db):
    from exercises import exercise_4_audit as ex4
    patch_module(ex4, gh)
    tid, result = asyncio.run(_run_ex4(PR3, lambda p: None))
    assert result["final_action"] == "auto_committed"
    ev = rows(db, tid)
    assert [e["action"] for e in ev] == ["fetch_pr", "analyze", "route", "auto_approve"]
    assert ev[-1]["decision"] == "auto" and ev[-1]["risk_level"] == "low"
    assert ev[-1]["reviewer_id"] is None
    assert len(gh.posted) == 1


def test_ex4_human_approval_audit_no_duplicate_pending(gh, db):
    from exercises import exercise_4_audit as ex4
    patch_module(ex4, gh)
    tid, result = asyncio.run(_run_ex4(PR1, lambda p: {"choice": "approve", "feedback": ""}))
    assert result["final_action"] == "committed"
    ev = rows(db, tid)
    assert [(e["action"], e["decision"]) for e in ev] == [
        ("fetch_pr", "pending"), ("analyze", "pending"), ("route", "pending"),
        ("human_approval", "pending"), ("human_approval", "approve"), ("commit", "approve"),
    ]
    assert ev[4]["reviewer_id"] == "LonelyStar05"
    assert {e["risk_level"] for e in ev[1:]} == {"med"}
    assert all(e["execution_time_ms"] >= 0 for e in ev)


def test_ex4_edit_rewrites_review(gh, db):
    from exercises import exercise_4_audit as ex4
    patch_module(ex4, gh)
    tid, result = asyncio.run(_run_ex4(
        PR1, lambda p: {"choice": "edit", "feedback": "Ask for an ALTER TABLE migration", "reviewer_id": "alice"}))
    assert result["final_action"] == "committed_with_edit"
    assert "[edited] Ask for an ALTER TABLE migration" in gh.posted[0][1]
    ev = rows(db, tid)
    assert ev[4]["reviewer_id"] == "alice" and ev[4]["decision"] == "edit"
    assert ev[-1]["action"] == "commit" and ev[-1]["decision"] == "edit"


def test_ex4_reject_posts_nothing(gh, db):
    from exercises import exercise_4_audit as ex4
    patch_module(ex4, gh)
    tid, result = asyncio.run(_run_ex4(PR1, lambda p: {"choice": "reject", "feedback": "wrong"}))
    assert result["final_action"] == "rejected" and gh.posted == []
    assert rows(db, tid)[-1]["decision"] == "reject"


def test_ex4_escalation_audit(gh, db):
    from exercises import exercise_4_audit as ex4
    patch_module(ex4, gh)
    tid, result = asyncio.run(_run_ex4(PR2, lambda p: {q: "yes" for q in p["questions"]}))
    assert result["final_action"] == "committed_after_escalation"
    ev = rows(db, tid)
    assert [(e["action"], e["decision"]) for e in ev] == [
        ("fetch_pr", "pending"), ("analyze", "pending"), ("route", "escalate"),
        ("escalate", "pending"), ("escalate", "escalate"), ("synthesize", "pending"),
        ("commit", "escalate"),
    ]
    assert ev[2]["risk_level"] == "high"
    assert ev[5]["confidence"] > ev[1]["confidence"]           # refined > initial
    # The README's sanity query: high risk must never be auto-committed.
    con = sqlite3.connect(db)
    assert con.execute(
        "SELECT COUNT(*) FROM audit_events WHERE risk_level='high' AND decision='auto'").fetchone()[0] == 0


def test_ex4_resume_after_process_restart(gh, db):
    """Pause at the interrupt, drop the app/checkpointer, then resume from SQLite."""
    from exercises import exercise_4_audit as ex4
    patch_module(ex4, gh)
    tid = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": tid}}

    async def first():
        async with open_checkpointer() as cp:
            r = await ex4.build_graph(cp).ainvoke({"pr_url": PR1, "thread_id": tid}, cfg)
            assert "__interrupt__" in r

    async def second():
        async with open_checkpointer() as cp:
            app = ex4.build_graph(cp)
            snap = await app.aget_state(cfg)
            assert snap.interrupts and snap.next == ("human_approval",)
            return await app.ainvoke(Command(resume={"choice": "approve"}), cfg)

    asyncio.run(first())
    result = asyncio.run(second())
    assert result["final_action"] == "committed"
    assert len(gh.posted) == 1
    assert sum(e["action"] == "human_approval" and e["decision"] == "pending" for e in rows(db, tid)) == 1


def test_replay_and_recent_threads(gh, db, capsys):
    from exercises import exercise_4_audit as ex4
    from audit import replay
    patch_module(ex4, gh)
    tid, _ = asyncio.run(_run_ex4(PR2, lambda p: {q: "ok" for q in p["questions"]}))
    threads = asyncio.run(replay.fetch_recent_threads())
    assert threads[0]["thread_id"] == tid
    assert threads[0]["worst_risk"] == "high"          # not 'med' (string MAX bug)
    asyncio.run(replay.replay(tid))
    out = capsys.readouterr().out
    assert "synthesize" in out and "commit" in out


@pytest.mark.parametrize("final_choice,expected,posted", [
    ("approve", "committed_after_escalation", 1), ("reject", "rejected", 0),
])
def test_ex4_confirm_refined_path_used_by_ui(gh, db, final_choice, expected, posted):
    """escalate → synthesize → human_approval(refined) → commit (Streamlit flow)."""
    from exercises import exercise_4_audit as ex4
    patch_module(ex4, gh)
    tid = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": tid}}
    kinds = []

    async def go():
        async with open_checkpointer() as cp:
            app = ex4.build_graph(cp, confirm_refined=True)
            r = await app.ainvoke({"pr_url": PR2, "thread_id": tid}, cfg)
            while "__interrupt__" in r:
                p = r["__interrupt__"][0].value
                kinds.append(p["kind"])
                if p["kind"] == "escalation":
                    resume = {"answers": {q: "yes" for q in p["questions"]}, "reviewer_id": "bob"}
                else:
                    assert p["refined"] and p["initial_confidence"] == 0.45 and p["confidence"] == 0.82
                    resume = {"choice": final_choice, "reviewer_id": "bob"}
                r = await app.ainvoke(Command(resume=resume), cfg)
            return r

    result = asyncio.run(go())
    assert kinds == ["escalation", "approval_request"]
    assert result["final_action"] == expected and len(gh.posted) == posted
    ev = rows(db, tid)
    assert [e["action"] for e in ev] == [
        "fetch_pr", "analyze", "route", "escalate", "escalate", "synthesize",
        "human_approval", "human_approval", "commit",
    ]
    assert {e["reviewer_id"] for e in ev if e["decision"] not in ("pending",) and e["action"] in ("escalate", "human_approval")} == {"bob"}
