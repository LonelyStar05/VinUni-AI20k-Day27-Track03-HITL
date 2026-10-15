"""Prompt builders shared by every exercise and the Streamlit app.

Kept in one place so the analyze / synthesize prompts stay identical across
exercise 1 → 4 and `app.py` (otherwise the confidence calibration drifts
between files and the same PR can land in different branches).
"""

from __future__ import annotations

from typing import Mapping

from common.schemas import PRAnalysis

MAX_DIFF_CHARS = 60_000  # keep the prompt bounded on huge PRs


ANALYZE_SYSTEM_PROMPT = """You are a senior software engineer doing a pull-request review.
Read the PR title, the list of changed files and the unified diff, then return a structured review.

- summary: one paragraph describing what the PR does.
- risk_factors: concrete risks (security, data loss, migrations, missing tests, breaking API changes).
- comments: actionable review comments, each tied to a file (and line when you can tell it from the
  diff hunk). Use severity: nit < suggestion < issue < blocker.
- confidence (0.0-1.0): how confident you are that THIS review is complete and correct, i.e. that a
  human reviewer would not find anything important you missed or misjudged. Calibrate honestly:
    * >= 0.80   trivial / mechanical change, you fully understand it, no open questions.
    * 0.60-0.72 small feature, mostly clear, but one or two points depend on context you do not
                have (e.g. schema migration strategy, intended behaviour).
    * < 0.55    security-sensitive code (auth, passwords, tokens, crypto, SQL, network sync),
                several red flags, or intent you cannot infer from the diff alone.
- confidence_reasoning: 1-3 sentences explaining the number.
- escalation_questions: if confidence < 0.60, write 2-4 specific, context-rich questions for the
  human reviewer that would resolve your uncertainty. Each question must reference the file /
  function / line in the diff it is about (e.g. "In auth.py `hash_password`, why MD5 instead of
  bcrypt/argon2?"). Leave the list empty when confidence >= 0.60.
"""


SYNTHESIZE_SYSTEM_PROMPT = """You are a senior software engineer finishing a pull-request review.
You already produced an initial review but were unsure, so a human reviewer answered your questions.
Produce a REFINED structured review that:

- incorporates the reviewer's answers as ground truth about intent and context;
- keeps every finding that is still valid (do not drop security issues just because the author
  meant well), removes findings the answers resolved, and adds any new ones the answers reveal;
- re-scores confidence: it should normally be HIGHER than the initial score because the open
  questions are now answered — but stay honest if the answers raise new concerns;
- leaves escalation_questions empty unless something critical is still unknown.
"""


def _clip(diff: str) -> str:
    if len(diff) > MAX_DIFF_CHARS:
        return diff[:MAX_DIFF_CHARS] + "\n... [diff truncated] ..."
    return diff


def build_review_messages(state: Mapping) -> list[dict]:
    """Messages for the first-pass `analyze` node."""
    files = "\n".join(f"- {f}" for f in state.get("pr_files") or [])
    return [
        {"role": "system", "content": ANALYZE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Title: {state.get('pr_title', '')}\n\n"
            f"Files changed:\n{files or '(unknown)'}\n\n"
            f"Diff:\n{_clip(state['pr_diff'])}"
        )},
    ]


def build_synthesis_messages(
    state: Mapping,
    initial: PRAnalysis,
    answers: Mapping[str, str],
) -> list[dict]:
    """Messages for the `synthesize` node (after escalation Q&A)."""
    qa = "\n\n".join(f"Q: {q}\nA: {a or '(no answer)'}" for q, a in answers.items())
    return [
        {"role": "system", "content": SYNTHESIZE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Title: {state.get('pr_title', '')}\n\n"
            f"Diff:\n{_clip(state['pr_diff'])}\n\n"
            f"--- Initial review (confidence {initial.confidence:.2f}) ---\n"
            f"{initial.model_dump_json(indent=2)}\n\n"
            f"--- Reviewer answers ---\n{qa}"
        )},
    ]


EDIT_SYSTEM_PROMPT = """You are a senior software engineer revising your own pull-request review.
A human reviewer read your review and asked for changes. Rewrite the review so it follows the
reviewer's instructions exactly (add / drop / reword comments, change severity, adjust the summary).
Keep everything the reviewer did not ask to change. Keep the same confidence unless the edit
clearly changes how sure you should be. Leave escalation_questions empty.
"""


def build_edit_messages(state: Mapping, current: PRAnalysis, feedback: str) -> list[dict]:
    """Messages for the bonus 'auto-edit' step (reviewer chose Edit)."""
    return [
        {"role": "system", "content": EDIT_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Title: {state.get('pr_title', '')}\n\n"
            f"Diff:\n{_clip(state['pr_diff'])}\n\n"
            f"--- Your current review ---\n{current.model_dump_json(indent=2)}\n\n"
            f"--- Reviewer's requested edits ---\n{feedback}"
        )},
    ]
