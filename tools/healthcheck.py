"""Project healthcheck for AIOS-Local."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_MODULES = [
    "fastapi",
    "uvicorn",
    "httpx",
    "numpy",
    "sklearn",
    "bs4",
    "duckduckgo_search",
    "apscheduler",
    "playwright",
]


def check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    return ok, f"Python {v.major}.{v.minor}.{v.micro}"


def check_modules() -> tuple[bool, str]:
    missing = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    if missing:
        return False, f"Missing modules: {', '.join(missing)}"
    return True, "All required modules import successfully"


def check_paths() -> tuple[bool, str]:
    expected = [
        ROOT / "agent.py",
        ROOT / "server.py",
        ROOT / "requirements.txt",
        ROOT / "memory",
        ROOT / "knowledge",
        ROOT / "config",
    ]
    missing = [str(p.relative_to(ROOT)) for p in expected if not p.exists()]
    if missing:
        return False, f"Missing paths: {', '.join(missing)}"
    return True, "Required project paths exist"


async def check_ollama() -> tuple[bool, str]:
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            r = await client.get("http://localhost:11434/api/tags")
        if r.status_code == 200:
            return True, "Ollama API reachable"
        return False, f"Ollama returned status {r.status_code}"
    except Exception as e:
        return False, f"Ollama not reachable: {e}"


def check_workspace_write() -> tuple[bool, str]:
    test_file = ROOT / "memory" / ".healthcheck.tmp"
    try:
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("ok", encoding="utf-8")
        _ = test_file.read_text(encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return True, "Workspace read/write OK"
    except Exception as e:
        return False, f"Workspace read/write failed: {e}"


async def main() -> int:
    checks = []

    checks.append(("python",) + check_python())
    checks.append(("modules",) + check_modules())
    checks.append(("paths",) + check_paths())
    checks.append(("workspace_io",) + check_workspace_write())
    checks.append(("ollama",) + await check_ollama())

    failed = 0
    print("=== AIOS-Local Healthcheck ===")
    for name, ok, msg in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {msg}")
        if not ok:
            failed += 1

    print("==============================")
    if failed:
        print(f"Healthcheck failed: {failed} check(s) failed")
        return 1
    print("Healthcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
