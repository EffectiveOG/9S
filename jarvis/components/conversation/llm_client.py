# jarvis/components/conversation/llm_client.py
"""Minimal async client for a local Ollama server.

Ported from Ciel's conversation module, but rewritten for asyncio: it talks to
Ollama's HTTP API with aiohttp (already a Jarvis dependency) instead of the
blocking `ollama` Python package, so generation never stalls the event loop.
"""

from typing import Any, Dict, List, Optional

from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM backend is unreachable or returns an error."""


class OllamaClient:
    """Chat with a model served by Ollama (https://ollama.com)."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            import aiohttp  # imported lazily so the package imports without it
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def _request(self, method: str, path: str,
                       payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        try:
            async with session.request(method, url, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise LLMError(f"Ollama {path} returned {resp.status}: {body[:200]}")
                return await resp.json()
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Ollama request to {url} failed: {e}") from e

    async def list_models(self) -> List[str]:
        data = await self._request("GET", "/api/tags")
        return [m.get("name", "") for m in data.get("models", [])]

    async def pull(self, model: str) -> None:
        logger.info(f"Pulling Ollama model {model} (this can take a while)...")
        await self._request("POST", "/api/pull", {"model": model, "stream": False})

    async def chat(self, model: str, messages: List[Dict[str, Any]],
                   options: Optional[Dict[str, Any]] = None) -> str:
        payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if options:
            payload["options"] = options
        data = await self._request("POST", "/api/chat", payload)
        return (data.get("message") or {}).get("content", "").strip()

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
