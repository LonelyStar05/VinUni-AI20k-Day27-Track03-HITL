"""Run the Streamlit UI fully offline (fake LLM + fake GitHub).

    HITL_DB_PATH=/tmp/demo.db uv run streamlit run tests/mock_app.py

Fake PRs: .../pull/1 → human_approval, /pull/2 → escalate, anything else → auto_approve.
Useful for UI demos/screenshots and for the Playwright E2E check.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exercises import exercise_4_audit as ex4  # noqa: E402
from tests.fakes import FakeGitHub, patch_module  # noqa: E402

_gh = getattr(ex4, "_mock_gh", None) or FakeGitHub()
ex4._mock_gh = _gh
patch_module(ex4, _gh)

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
