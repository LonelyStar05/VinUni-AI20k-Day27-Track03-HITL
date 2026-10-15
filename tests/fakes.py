"""Offline fakes for the LLM and GitHub so the whole graph runs without network.

Used by the pytest suite and by `tests/mock_app.py` (Streamlit UI in mock mode).

The fake LLM is deterministic and keyed off the diff content, mirroring the
behaviour the README expects from the two demo PRs:

    diff mentions "md5"/"password"  → 0.45  (escalate, with questions)
    diff mentions "priority"        → 0.65  (human_approval)
    anything else                   → 0.90  (auto_approve)

The synthesize prompt always returns 0.82; the edit prompt applies the
reviewer's feedback as an extra comment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from common.github import PullRequest, parse_pr_url
from common.schemas import PRAnalysis, ReviewComment


PR1_DIFF = """diff --git a/todo/models.py b/todo/models.py
--- a/todo/models.py
+++ b/todo/models.py
@@ -8,6 +8,7 @@ class Task:
     id: int
     title: str
     done: bool = False
+    priority: str = "medium"   # low | medium | high
diff --git a/todo/storage.py b/todo/storage.py
--- a/todo/storage.py
+++ b/todo/storage.py
@@ -12,7 +12,7 @@ class Storage:
-        self.conn.execute("CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY, title TEXT, done INT)")
+        self.conn.execute("CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY, title TEXT, done INT, priority TEXT)")
"""

PR2_DIFF = """diff --git a/todo/auth.py b/todo/auth.py
new file mode 100644
--- /dev/null
+++ b/todo/auth.py
@@ -0,0 +1,14 @@
+import hashlib
+SYNC_URL = "http://sync.example.com/api"
+def hash_password(pw):
+    return hashlib.md5(pw.encode()).hexdigest()
+def save_token(token):
+    open("token.txt", "w").write(token)
diff --git a/todo/storage.py b/todo/storage.py
--- a/todo/storage.py
+++ b/todo/storage.py
@@ -20,3 +20,3 @@ class Storage:
-        self.conn.execute("INSERT INTO tasks (title) VALUES (?)", (title,))
+        self.conn.execute(f"INSERT INTO tasks (title, user_id) VALUES ('{title}', 1)")
"""

PR3_DIFF = """diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1,3 +1,3 @@
-# Todo ap
+# Todo app
"""

FAKE_PRS = {
    1: ("Add task priority field", PR1_DIFF, ["todo/models.py", "todo/storage.py"]),
    2: ("Add user login + cloud sync", PR2_DIFF, ["todo/auth.py", "todo/storage.py"]),
    3: ("Fix typo in README", PR3_DIFF, ["README.md"]),
}

PR2_QUESTIONS = [
    "In todo/auth.py `hash_password`, why MD5 instead of bcrypt/argon2?",
    "Is `SYNC_URL` in todo/auth.py meant to be HTTPS in production?",
    "In todo/storage.py `Storage.add`, why switch from a parameterised query to an f-string?",
]


def fake_fetch_pr(pr_url: str) -> PullRequest:
    owner, repo, number = parse_pr_url(pr_url)
    title, diff, files = FAKE_PRS.get(number, FAKE_PRS[3])
    return PullRequest(
        url=pr_url, owner=owner, repo=repo, number=number, title=title,
        author="student", base_ref="main", head_ref=f"feature-{number}",
        head_sha=f"{number:07d}abcdef", diff=diff, files_changed=files,
    )


@dataclass
class FakeGitHub:
    posted: list[tuple[str, str]] = field(default_factory=list)

    def post_review_comment(self, pr_url: str, body: str) -> str:
        self.posted.append((pr_url, body))
        return f"{pr_url}#issuecomment-{len(self.posted)}"


def _analysis_for(messages: list[dict]) -> PRAnalysis:
    system = messages[0]["content"]
    user = messages[-1]["content"]

    if "finishing a pull-request review" in system:          # synthesize
        return PRAnalysis(
            summary="Adds login + cloud sync. Reviewer confirmed MD5 is a placeholder; must ship with bcrypt.",
            risk_factors=["MD5 password hashing", "SQL injection in Storage.add"],
            comments=[
                ReviewComment(file="todo/auth.py", line=4, severity="blocker",
                              body="Replace MD5 with bcrypt/argon2 before merge."),
                ReviewComment(file="todo/storage.py", line=21, severity="blocker",
                              body="Use a parameterised query; the f-string is injectable."),
            ],
            confidence=0.82,
            confidence_reasoning="Reviewer answers resolved intent; remaining issues are clear blockers.",
        )

    if "revising your own pull-request review" in system:    # edit (bonus 4)
        feedback = user.split("--- Reviewer's requested edits ---", 1)[-1].strip()
        return PRAnalysis(
            summary="Adds a priority field to tasks (edited per reviewer).",
            comments=[ReviewComment(file="todo/storage.py", severity="issue", body=f"[edited] {feedback}")],
            confidence=0.66,
            confidence_reasoning="Edited per reviewer feedback.",
        )

    low = user.lower()
    if "md5" in low or "password" in low:
        return PRAnalysis(
            summary="Adds user login and cloud sync.",
            risk_factors=["MD5 password hashing", "Plaintext token storage",
                          "SQL injection in Storage.add", "Hard-coded user_id", "No tests"],
            comments=[ReviewComment(file="todo/auth.py", line=4, severity="blocker", body="MD5 is broken for passwords.")],
            confidence=0.45,
            confidence_reasoning="Security-sensitive change with several red flags and unclear intent.",
            escalation_questions=PR2_QUESTIONS,
        )
    if "priority" in low:
        return PRAnalysis(
            summary="Adds a `priority` field to Task and the tasks table.",
            risk_factors=["Existing DBs need a migration for the new column"],
            comments=[ReviewComment(file="todo/storage.py", line=13, severity="issue",
                                    body="CREATE TABLE IF NOT EXISTS won't add the column to existing databases.")],
            confidence=0.65,
            confidence_reasoning="Small change, but the migration strategy is unclear.",
        )
    return PRAnalysis(
        summary="Fixes a typo in the README.",
        comments=[],
        confidence=0.90,
        confidence_reasoning="Trivial documentation change.",
    )


class _Structured:
    def invoke(self, messages):
        return _analysis_for(messages)

    async def ainvoke(self, messages):
        return _analysis_for(messages)


class FakeLLM:
    def with_structured_output(self, schema):
        assert schema is PRAnalysis
        return _Structured()


def fake_get_llm(*_a, **_kw) -> FakeLLM:
    return FakeLLM()


def patch_module(mod, gh: FakeGitHub) -> None:
    """Swap network-touching names inside an exercise module."""
    for name, value in [
        ("get_llm", fake_get_llm),
        ("fetch_pr", fake_fetch_pr),
        ("post_review_comment", gh.post_review_comment),
    ]:
        if hasattr(mod, name):
            setattr(mod, name, value)
