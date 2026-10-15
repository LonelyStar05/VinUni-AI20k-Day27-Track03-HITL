"""Replay a review session from the audit trail.

Usage:
    uv run python -m audit.replay --thread <thread_id>
    uv run python -m audit.replay --list                # list recent threads

The script reads `audit_events` (human-readable timeline). The LangGraph
SqliteSaver checkpoint tables live in the same .db file but are queried
separately by the LangGraph runtime, not by this tool.
"""

from __future__ import annotations

import argparse
import asyncio

from rich.console import Console
from rich.table import Table

from common.db import db_conn, replay_events


RISK_COLOR = {"low": "green", "med": "yellow", "high": "red"}


async def fetch_recent_threads(limit: int = 25) -> list[dict]:
    """Recent sessions as plain dicts (used by the CLI table and the Streamlit sidebar).

    `worst_risk` ranks high > med > low explicitly — a plain MAX(risk_level)
    compares strings, which would rank 'med' above 'high'. Rows written by
    `fetch_pr` (no confidence yet, neutral 'med') are excluded from the
    ranking so they don't mask the real outcome.
    """
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp)        AS started,
                   MAX(timestamp)        AS last_event,
                   CASE MAX(CASE WHEN action = 'fetch_pr' THEN 0
                                 WHEN risk_level = 'high' THEN 3
                                 WHEN risk_level = 'med'  THEN 2
                                 WHEN risk_level = 'low'  THEN 1 ELSE 0 END)
                        WHEN 3 THEN 'high' WHEN 2 THEN 'med' WHEN 1 THEN 'low'
                        ELSE '-' END     AS worst_risk,
                   (SELECT e2.decision FROM audit_events e2
                     WHERE e2.thread_id = audit_events.thread_id
                     ORDER BY e2.id DESC LIMIT 1) AS last_decision,
                   (SELECT e3.action FROM audit_events e3
                     WHERE e3.thread_id = audit_events.thread_id
                     ORDER BY e3.id DESC LIMIT 1) AS last_action,
                   COUNT(*)              AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def list_threads() -> None:
    console = Console()
    rows = await fetch_recent_threads()

    table = Table(title="Recent review sessions")
    for col in ("thread_id", "pr_url", "started", "last_event", "worst_risk", "last_action", "events"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r["thread_id"],
            r["pr_url"],
            str(r["started"]),
            str(r["last_event"]),
            str(r["worst_risk"]),
            str(r["last_action"]),
            str(r["events"]),
        )
    console.print(table)


async def replay(thread_id: str) -> None:
    console = Console()
    events = await replay_events(thread_id)

    if not events:
        console.print(f"[red]No events found for thread {thread_id}[/red]")
        return

    console.rule(f"[bold]Replay {thread_id}")
    for ev in events:
        risk = ev["risk_level"]
        risk_colored = f"[{RISK_COLOR.get(risk, 'white')}]{risk:<4}[/]"
        reviewer = ev["reviewer_id"] or "-"
        reason = (ev["reason"] or "")[:60]
        console.print(
            f"[dim]{ev['timestamp']}[/dim]  "
            f"[cyan]{ev['action']:<18}[/cyan] "
            f"conf=[bold]{ev['confidence']:.2f}[/bold] "
            f"risk={risk_colored} "
            f"decision=[magenta]{ev['decision']:<10}[/magenta] "
            f"reviewer={reviewer:<14} "
            f"{ev['execution_time_ms']:>5}ms  "
            f"[dim]{reason}[/dim]"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", help="Replay one thread by id")
    parser.add_argument("--list", action="store_true", help="List recent threads")
    args = parser.parse_args()

    if args.list:
        asyncio.run(list_threads())
    elif args.thread:
        asyncio.run(replay(args.thread))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
