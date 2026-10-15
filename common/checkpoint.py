"""Durable LangGraph checkpointer shared by exercise 4 and the Streamlit app.

`PRAnalysis` / `ReviewComment` (Pydantic) live in the graph state, so they are
msgpack-serialized into checkpoints. Recent langgraph-checkpoint versions warn
(and will soon refuse) when deserializing types that are not explicitly
allow-listed, so we register ours here.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from common.db import db_path

ALLOWED_STATE_TYPES = [
    ("common.schemas", "PRAnalysis"),
    ("common.schemas", "ReviewComment"),
]


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """Open the SQLite checkpointer (same file as audit_events) and run setup()."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        cp.serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_STATE_TYPES)
        await cp.setup()
        yield cp
