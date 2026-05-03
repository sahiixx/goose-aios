"""Web scraping utilities and safety wrappers."""

import asyncio
import importlib
import re
from pathlib import Path
from typing import Optional

from .config import (
    ALLOWED_WEB_DOMAINS,
    BASE_DIR,
    BASH_ALLOWED_PREFIXES,
    DESTRUCTIVE_RE,
    _SENSITIVE_FILENAMES,
)
from .safety import (
    blocked_by_mode,
    command_policy_block_reason,
    safe_resolve_path,
    validate_write_target,
)

try:
    import httpx  # pyright: ignore[reportMissingImports]
    _ConnectError = httpx.ConnectError
except ImportError:
    httpx = None
    class _ConnectError(Exception):
        pass

try:
    import numpy as np  # pyright: ignore[reportMissingImports]
except ImportError:
    np = None

try:
    from bs4 import BeautifulSoup  # pyright: ignore[reportMissingImports]
except ImportError:
    BeautifulSoup = None

try:
    from duckduckgo_search import DDGS  # pyright: ignore[reportMissingImports]
except ImportError:
    DDGS = None


def _blocked_by_mode(tool_name: str) -> Optional[str]:
    from .config import (
        HIGH_RISK_TOOL_SET,
        REQUIRE_HIGH_RISK_APPROVAL,
        SAFETY_MODE,
        _HIGH_RISK_REASON,
    )
    return blocked_by_mode(
        tool_name, SAFETY_MODE, HIGH_RISK_TOOL_SET, REQUIRE_HIGH_RISK_APPROVAL, _HIGH_RISK_REASON
    )


def _safe_resolve_path(path_value: str) -> Path:
    return safe_resolve_path(path_value, BASE_DIR)


def _validate_write_target(path: Path) -> Optional[str]:
    return validate_write_target(path, _SENSITIVE_FILENAMES)


def _command_policy_block_reason(cmd: str) -> Optional[str]:
    return command_policy_block_reason(cmd, DESTRUCTIVE_RE, BASH_ALLOWED_PREFIXES)


def _domain_allowed(url: str) -> bool:
    if not ALLOWED_WEB_DOMAINS:
        return True
    if httpx is None:
        return False
    try:
        host = httpx.URL(url).host or ""
    except Exception:
        return False
    return any(host == d or host.endswith(f".{d}") for d in ALLOWED_WEB_DOMAINS)


async def crawl_url(url: str, request_timeout_sec: int = 20) -> dict:
    if not _domain_allowed(url):
        return {"url": url, "error": "domain_not_allowed"}
    if httpx is None:
        return {"url": url, "error": "httpx_not_available"}
    try:
        async with httpx.AsyncClient(timeout=request_timeout_sec, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "AIOS-Local/3.0"})
            r.raise_for_status()
        html = r.text
        if BeautifulSoup is None:
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return {"url": url, "title": "", "text": text[:12000]}

        soup = BeautifulSoup(html, "lxml")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return {"url": url, "title": title, "text": text[:12000]}
    except Exception as e:
        return {"url": url, "error": str(e)}


async def web_search(query: str) -> str:
    if not query.strip():
        return "Search error: empty query"
    if DDGS is None:
        return "Search error: duckduckgo-search is not installed"

    try:
        results = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=10):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", ""),
                    }
                )

        if not results:
            return "No results"

        crawled = await asyncio.gather(
            *(crawl_url(item["url"]) for item in results[:5] if item.get("url")),
            return_exceptions=True,
        )

        lines = []
        for i, item in enumerate(results[:8], 1):
            lines.append(f"[{i}] {item['title']}\n{item['snippet']}\n{item['url']}")

        excerpts = []
        for page in crawled:
            if isinstance(page, dict) and page.get("text"):
                title = page.get("title", "")
                excerpts.append(f"{title}\n{page['url']}\n{page['text'][:350]}")

        if excerpts:
            lines.append("\nCrawled Excerpts:\n" + "\n\n".join(excerpts[:3]))
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


async def browser_extract(url: str) -> str:
    if not _domain_allowed(url):
        return "Blocked: domain not allowed"
    try:
        async_playwright = importlib.import_module("playwright.async_api").async_playwright
    except Exception:
        return "Browser error: playwright is not installed"

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            text = await page.inner_text("body")
            await browser.close()
            return text[:12000]
    except Exception as e:
        return f"Browser error: {e}"
