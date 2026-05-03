"""Telemetry event logger."""

import json
from datetime import datetime, timezone

from .config import TELEMETRY_FILE


def _telemetry(event: str, **fields):
    payload = {
        "event": event,
        "ts": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    try:
        with TELEMETRY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return
