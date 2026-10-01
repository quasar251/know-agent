"""know retrieval evaluation harness.

Standalone, offline-capable tooling that measures the *retrieval* half of the
RAG pipeline (quality + latency) for a given KB. It deliberately re-implements
KBSearchTool's pipeline against the same infra layer (`src.infra.*`) instead of
calling the tool, so we can keep the full ordered hit list — KBSearchTool's
formatted `raw["sources"]` is sorted by cosine score and loses the rerank order.

Importing this package puts `backend/` on sys.path so `src.*` resolves.
"""
from __future__ import annotations

import sys
from pathlib import Path

# eval/__init__.py → repo root is parents[1], backend is root/backend
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = _REPO_ROOT / "backend"

for _p in (str(_REPO_ROOT), str(_BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)