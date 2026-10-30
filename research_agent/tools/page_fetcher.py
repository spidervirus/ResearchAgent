"""
PageFetcherTool — fetches a URL and returns clean, readable text content.

Pipeline:
  1. httpx GET with browser-like headers (15 s timeout)
  2. trafilatura extracts main article text (best-in-class for this)
  3. BeautifulSoup fallback if trafilatura returns nothing
  4. Truncate to max_chars
  5. Return structured dict with url, title, content, word_count, fetch_time_ms

Handles: timeouts, HTTP errors, binary/PDF responses, oversized pages.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional
from urllib.parse import urlparse

import httpx

from research_agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_DEFAULT_HEADERS = {
    "User-Agent": _DEFAULT_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Content-types we consider fetchable HTML
_HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "text/xhtml"}

# Very rough upper bound for page download (bytes) before we stop reading
_MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

# CSS selectors / tag names we strip out in the BS fallback
_BOILERPLATE_TAGS = [
    "nav", "header", "footer", "aside", "script", "style",
    "noscript", "iframe", "form", "button", "svg",
]
_BOILERPLATE_CLASSES_RE = re.compile(
    r"(cookie|banner|popup|modal|overlay|advertisement|ad[-_]|sidebar|nav[-_]|"
    r"menu|header|footer|related|social|share|promo|subscribe|newsletter)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helper: trafilatura extraction
# ---------------------------------------------------------------------------


def _extract_with_trafilatura(html: str, url: str) -> Optional[str]:
    """Use trafilatura to extract main content. Returns None on failure."""
    try:
        import trafilatura  # type: ignore[import]

        result = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        return result or None
    except Exception:  # noqa: BLE001
        logger.debug("trafilatura extraction failed for %s", url, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Helper: BeautifulSoup fallback
# ---------------------------------------------------------------------------


def _extract_with_bs4(html: str) -> tuple[str, str]:
    """
    Fallback text extraction using BeautifulSoup.
    Returns (title, body_text).
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "beautifulsoup4 is not installed. Run: pip install beautifulsoup4"
        ) from exc

    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Remove boilerplate elements by tag
    for tag_name in _BOILERPLATE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove elements whose class/id hints at boilerplate
    for tag in soup.find_all(True):
        class_str = " ".join(tag.get("class", []))
        id_str = tag.get("id", "")
        if _BOILERPLATE_CLASSES_RE.search(class_str) or _BOILERPLATE_CLASSES_RE.search(id_str):
            tag.decompose()

    # Prefer <article> > <main> > <body>
    for selector in ["article", "main", "body"]:
        target = soup.find(selector)
        if target:
            text = target.get_text(separator="\n", strip=True)
            if len(text) > 200:
                return title, text

    return title, soup.get_text(separator="\n", strip=True)


# ---------------------------------------------------------------------------
# Helper: extract title from trafilatura metadata
# ---------------------------------------------------------------------------


def _extract_title_trafilatura(html: str, url: str) -> str:
    """Best-effort title extraction using trafilatura metadata."""
    try:
        import trafilatura  # type: ignore[import]
        from trafilatura.metadata import extract_metadata  # type: ignore[import]

        meta = extract_metadata(html, default_url=url)
        if meta and meta.title:
            return meta.title
    except Exception:  # noqa: BLE001
        pass
    return ""


# ---------------------------------------------------------------------------
# Helper: normalise whitespace
# ---------------------------------------------------------------------------


def _normalise_whitespace(text: str) -> str:
    # Collapse 3+ blank lines to 2, strip trailing spaces on each line
    lines = [line.rstrip() for line in text.splitlines()]
    out: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                out.append("")
        else:
            blank_run = 0
            out.append(line)
    return "\n".join(out).strip()


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------


class PageFetcherTool(BaseTool):
    """
    Fetches a web page and returns clean, readable text.

    Usage::

        tool = PageFetcherTool()
        result = await tool.run("https://example.com/article", max_chars=5000)
        print(result.data["content"])
    """

    name = "page_fetcher"
    description = (
        "Fetch a web page and extract its main text content, "
        "stripping navigation, ads, and other boilerplate."
    )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                        "max_chars": {"type": "integer", "description": "Maximum characters to extract (default 8000)"},
                    },
                    "required": ["url"],
                },
            },
        }

    def __init__(self, timeout: float = 15.0) -> None:
        self._timeout = timeout

    async def run(self, url: str, max_chars: int = 8000) -> ToolResult:  # type: ignore[override]
        """
        Fetch a URL and return extracted text.

        Args:
            url:       Fully-qualified URL to fetch.
            max_chars: Maximum characters of content to return (default 8 000).

        Returns:
            ToolResult where .data is a dict::

                {
                    "url":          str,   # final URL after redirects
                    "title":        str,
                    "content":      str,   # cleaned body text, truncated
                    "word_count":   int,
                    "fetch_time_ms": float,
                    "truncated":    bool,
                }
        """
        if not url or not url.strip():
            return self._err("URL must not be empty.")

        url = url.strip()

        # Basic URL sanity check
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return self._err(f"Unsupported URL scheme '{parsed.scheme}'. Only http/https allowed.")

        fetch_start = time.perf_counter()
        html: str = ""
        final_url: str = url
        content_type: str = ""
        status_code: int = 0

        # ------------------------------------------------------------------
        # 1. Fetch
        # ------------------------------------------------------------------
        try:
            async with httpx.AsyncClient(
                headers=_DEFAULT_HEADERS,
                timeout=self._timeout,
                follow_redirects=True,
                max_redirects=5,
            ) as client:
                async with client.stream("GET", url) as response:
                    status_code = response.status_code
                    final_url = str(response.url)
                    content_type = response.headers.get("content-type", "").lower()

                    # Handle HTTP errors
                    if status_code >= 400:
                        fetch_ms = (time.perf_counter() - fetch_start) * 1000
                        return self._err(
                            f"HTTP {status_code} fetching {url}",
                            data={"url": final_url, "status_code": status_code},
                            duration_ms=fetch_ms,
                        )

                    # Detect non-HTML content
                    content_type_base = content_type.split(";")[0].strip()
                    if content_type_base == "application/pdf":
                        fetch_ms = (time.perf_counter() - fetch_start) * 1000
                        return self._ok(
                            {
                                "url": final_url,
                                "title": "",
                                "content": "[PDF document — text extraction not supported]",
                                "word_count": 0,
                                "fetch_time_ms": round(fetch_ms, 2),
                                "truncated": False,
                            },
                            duration_ms=fetch_ms,
                        )

                    if content_type_base and not any(
                        ct in content_type_base for ct in ("html", "text", "xml")
                    ):
                        fetch_ms = (time.perf_counter() - fetch_start) * 1000
                        return self._err(
                            f"Non-text content-type '{content_type_base}' — skipping.",
                            data={"url": final_url, "content_type": content_type},
                            duration_ms=fetch_ms,
                        )

                    # Stream body with size cap
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        chunks.append(chunk)
                        total += len(chunk)
                        if total >= _MAX_DOWNLOAD_BYTES:
                            logger.warning(
                                "Page at %s exceeded %d bytes — truncating download.",
                                url,
                                _MAX_DOWNLOAD_BYTES,
                            )
                            break

                    raw_bytes = b"".join(chunks)

        except httpx.TimeoutException:
            return self._err(
                f"Request timed out after {self._timeout}s fetching {url}",
                duration_ms=(time.perf_counter() - fetch_start) * 1000,
            )
        except httpx.TooManyRedirects:
            return self._err(
                f"Too many redirects fetching {url}",
                duration_ms=(time.perf_counter() - fetch_start) * 1000,
            )
        except httpx.RequestError as exc:
            return self._err(
                f"Network error fetching {url}: {exc}",
                duration_ms=(time.perf_counter() - fetch_start) * 1000,
            )

        fetch_ms = (time.perf_counter() - fetch_start) * 1000

        # ------------------------------------------------------------------
        # 2. Decode
        # ------------------------------------------------------------------
        encoding = "utf-8"
        if "charset=" in content_type:
            enc_part = content_type.split("charset=")[-1].split(";")[0].strip()
            if enc_part:
                encoding = enc_part

        try:
            html = raw_bytes.decode(encoding, errors="replace")
        except LookupError:
            html = raw_bytes.decode("utf-8", errors="replace")

        if not html.strip():
            return self._err("Received empty response body.", duration_ms=fetch_ms)

        # ------------------------------------------------------------------
        # 3. Extract content — trafilatura first, BS4 fallback
        # ------------------------------------------------------------------
        title = _extract_title_trafilatura(html, final_url)
        content: Optional[str] = _extract_with_trafilatura(html, final_url)

        if not content or len(content.strip()) < 100:
            logger.debug("trafilatura returned little/nothing for %s — falling back to BS4", url)
            bs4_title, content = _extract_with_bs4(html)
            if not title:
                title = bs4_title

        if not content:
            content = ""

        # ------------------------------------------------------------------
        # 4. Normalise + truncate
        # ------------------------------------------------------------------
        content = _normalise_whitespace(content)
        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars].rsplit("\n", 1)[0]  # break at last newline

        word_count = len(content.split())

        logger.info(
            "PageFetcherTool: fetched %s → %d chars (%d words) in %.1fms",
            url, len(content), word_count, fetch_ms,
        )

        return self._ok(
            {
                "url": final_url,
                "title": title,
                "content": content,
                "word_count": word_count,
                "fetch_time_ms": round(fetch_ms, 2),
                "truncated": truncated,
            },
            duration_ms=fetch_ms,
        )
