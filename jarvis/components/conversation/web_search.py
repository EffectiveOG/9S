# jarvis/components/conversation/web_search.py
"""Web search for the conversation component, ported from Ciel.

Like Ciel's WebSearchModule it uses Google Custom Search when credentials are
available and falls back to DuckDuckGo, can extract page text, decides from
the question whether a search is useful, and condenses results into a prompt
context for the LLM.

Differences from Ciel:
  * asyncio-native: Google is called over plain HTTPS with aiohttp (no
    google-api-python-client dependency), DuckDuckGo's blocking client runs
    in a worker thread, page fetches run concurrently;
  * every backend is optional - a missing library or key just disables it;
  * search triggers are tightened (no "temps"/"quelle heure", which made Ciel
    search the web for the time of day) and exist in French and English.
"""

import asyncio
import os
import re
from typing import Any, Dict, List, Optional

from ...utils.logging_utils import get_logger

logger = get_logger(__name__)

GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_TRIGGERS = {
    "fr": {
        "words": [
            "actualité", "nouvelles", "infos", "dernières", "derniers", "récent",
            "aujourd'hui", "cette semaine", "ce mois", "cette année",
            "météo", "température", "prix", "cours", "bourse", "taux",
            "horaires", "programme", "événements", "spectacle", "résultat",
            "restaurant", "hôtel", "itinéraire", "sur internet", "sur le web",
            "recherche", "cherche", "trouve",
        ],
        "patterns": [
            r"que (se passe|s'est passé)", r"quoi de (neuf|nouveau)",
            r"(qui a gagné|qui est le président|qui est la présidente)",
            r"\b20\d\d\b",
        ],
    },
    "en": {
        "words": [
            "news", "latest", "recent", "today", "this week", "this month",
            "this year", "weather", "temperature", "price", "stock", "exchange rate",
            "schedule", "opening hours", "events", "score", "results",
            "restaurant", "hotel", "directions", "on the internet", "on the web",
            "search", "look up", "google",
        ],
        "patterns": [
            r"what('s| is) (happening|going on)", r"who won", r"who is the president",
            r"\b20\d\d\b",
        ],
    },
}


def _env_or_none(value: Optional[str]) -> Optional[str]:
    """Treat empty values and unexpanded "${VAR}" placeholders as unset."""
    if not value or str(value).startswith("$"):
        return None
    return str(value)


class WebSearch:
    """Google Custom Search with DuckDuckGo fallback."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, language: str = "fr"):
        config = config or {}
        self.language = language if language in _TRIGGERS else "en"
        self.provider = str(config.get("provider", "auto")).lower()
        self.max_results = int(config.get("max_results", 3))
        self.fetch_content = bool(config.get("fetch_content", False))
        self.max_content_length = int(config.get("max_content_length", 1500))
        self.timeout = float(config.get("timeout", 10))
        self.region = config.get("region", "fr-fr" if self.language == "fr" else "us-en")
        self.time_limit = config.get("time_limit", "m")  # d / w / m / y

        self.google_api_key = _env_or_none(config.get("google_api_key")) or _env_or_none(
            os.getenv("GOOGLE_API_KEY"))
        self.google_cse_id = _env_or_none(config.get("google_cse_id")) or _env_or_none(
            os.getenv("GOOGLE_CSE_ID"))

        self._session = None
        self._ddgs_cls = self._load_ddgs()

    # ------------------------------------------------------------ backends
    @staticmethod
    def _load_ddgs():
        try:
            from ddgs import DDGS  # current package name
            return DDGS
        except Exception:
            pass
        try:
            from duckduckgo_search import DDGS  # legacy package name (used by Ciel)
            return DDGS
        except Exception:
            return None

    @property
    def google_enabled(self) -> bool:
        return (self.provider in ("auto", "google")
                and bool(self.google_api_key and self.google_cse_id))

    @property
    def duckduckgo_enabled(self) -> bool:
        return self.provider in ("auto", "duckduckgo") and self._ddgs_cls is not None

    @property
    def available(self) -> bool:
        return self.google_enabled or self.duckduckgo_enabled

    async def _get_session(self):
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"User-Agent": _USER_AGENT},
            )
        return self._session

    async def _get_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        session = await self._get_session()
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _get_text(self, url: str) -> str:
        session = await self._get_session()
        async with session.get(url) as resp:
            resp.raise_for_status()
            if "html" not in resp.headers.get("Content-Type", "html"):
                return ""
            return await resp.text(errors="ignore")

    # ------------------------------------------------------------- search
    async def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search the web; returns [{title, url, snippet, content, date, source}]."""
        n = max_results or self.max_results
        results: List[Dict[str, Any]] = []
        if self.google_enabled:
            try:
                results = await self._search_google(query, n)
            except Exception as e:
                logger.warning(f"Google search failed, falling back: {e}")
        if not results and self.duckduckgo_enabled:
            try:
                results = await self._search_duckduckgo(query, n)
            except Exception as e:
                logger.warning(f"DuckDuckGo search failed: {e}")
        if results and self.fetch_content:
            await self._attach_content(results)
        logger.info(f"Web search '{query}': {len(results)} result(s)")
        return results

    async def _search_google(self, query: str, n: int) -> List[Dict[str, Any]]:
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": min(n, 10),
            "hl": self.language,
            "gl": self.region.split("-")[0],
        }
        if self.time_limit:
            params["dateRestrict"] = f"{self.time_limit}1"
        data = await self._get_json(GOOGLE_CSE_URL, params)
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "content": None,
                "date": None,
                "source": item.get("displayLink") or "google",
            }
            for item in data.get("items", [])[:n]
        ]

    async def _search_duckduckgo(self, query: str, n: int) -> List[Dict[str, Any]]:
        def run():
            ddgs = self._ddgs_cls()
            # positional query: `ddgs` names it `query`, `duckduckgo_search` `keywords`
            return list(ddgs.text(query, region=self.region, safesearch="moderate",
                                  timelimit=self.time_limit, max_results=n) or [])

        raw = await asyncio.to_thread(run)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href") or r.get("url", ""),
                "snippet": r.get("body", ""),
                "content": None,
                "date": r.get("date"),
                "source": r.get("source") or "duckduckgo",
            }
            for r in raw[:n]
        ]

    async def _attach_content(self, results: List[Dict[str, Any]]):
        async def fetch(result):
            url = result.get("url") or ""
            if not url.startswith(("http://", "https://")):
                return
            try:
                html = await self._get_text(url)
                text = self.extract_text(html)
                if text:
                    result["content"] = text[: self.max_content_length]
            except Exception as e:
                logger.debug(f"Content extraction failed for {url}: {e}")

        await asyncio.gather(*(fetch(r) for r in results))

    @staticmethod
    def extract_text(html: str) -> Optional[str]:
        """Main text of an HTML page (BeautifulSoup if installed, regex otherwise)."""
        if not html:
            return None
        text = ""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()
            for selector in ("main", "article", "#content", ".content", ".post", "#main"):
                node = soup.select_one(selector)
                if node:
                    text = node.get_text(" ", strip=True)
                    break
            if not text and soup.body:
                text = soup.body.get_text(" ", strip=True)
        except ImportError:
            stripped = re.sub(r"(?is)<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>", " ", html)
            text = re.sub(r"(?s)<[^>]+>", " ", stripped)
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) > 50 else None

    # ------------------------------------------------------------ helpers
    def is_search_query(self, text: str) -> bool:
        """Heuristic: does answering `text` need fresh information from the web?"""
        t = text.lower()
        triggers = _TRIGGERS[self.language]
        if any(w in t for w in triggers["words"]):
            return True
        return any(re.search(p, t) for p in triggers["patterns"])

    @staticmethod
    def summarize(results: List[Dict[str, Any]], max_items: int = 3) -> str:
        """Condense results into a plain-text prompt context for the LLM."""
        lines = []
        for i, r in enumerate(results[:max_items], 1):
            lines.append(f"{i}. {r.get('title', '').strip()}")
            if r.get("snippet"):
                lines.append(f"   {r['snippet'].strip()}")
            if r.get("content"):
                excerpt = " ".join(re.split(r"(?<=[.!?])\s+", r["content"][:600])[:3])
                lines.append(f"   Extract: {excerpt}")
            meta = ", ".join(x for x in (r.get("source"), r.get("date")) if x)
            if meta:
                lines.append(f"   ({meta})")
        return "\n".join(lines)

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
