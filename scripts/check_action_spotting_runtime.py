"""Local preflight check for KickClip Champion Action Spotting.

Run from the backend project root:
    python scripts/check_action_spotting_runtime.py

The command does not require a running FastAPI server or a database connection.
It validates the Champion model, PyTorch load/inference contract, SoccerNet
feature-extraction files/dependencies, and ffmpeg/ffprobe availability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domains.action_spotting.diagnostics import (  # noqa: E402
    collect_action_spotting_diagnostics,
)


def main() -> int:
    diagnostics = collect_action_spotting_diagnostics(db=None)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return 0 if diagnostics.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
