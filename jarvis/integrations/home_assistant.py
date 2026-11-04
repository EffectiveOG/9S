# jarvis/integrations/home_assistant.py
"""Async client for the Home Assistant WebSocket API.

One long-lived connection is used both to *observe* the house (a cache of every
entity state, kept current via `state_changed` events) and to *act* on it
(`call_service`). The client reconnects on its own; callers never deal with the
socket.

Protocol reference: https://developers.home-assistant.io/docs/api/websocket
"""

import asyncio
import itertools
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# (entity_id, old_state, new_state) -> None. States are HA state dicts or None.
StateListener = Callable[[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]], Awaitable[None]]


class HomeAssistantAuthError(Exception):
    """Raised when Home Assistant rejects the access token."""


def is_configured(value: Any) -> bool:
    """True if a config value was actually provided.

    JarvisCore expands ${VAR} from the environment, but os.path.expandvars
    leaves unset variables untouched, so "${HA_URL}" means "not configured".
    """
    return isinstance(value, str) and value.strip() != "" and "${" not in value


def websocket_url(base_url: str) -> str:
    """http://ha.local:8123 -> ws://ha.local:8123/api/websocket"""
    url = base_url.rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    if not url.endswith("/api/websocket"):
        url += "/api/websocket"
    return url


class HomeAssistantClient:
    """Keeps a live mirror of Home Assistant states and calls services."""

    def __init__(self, url: str, token: str, reconnect_delay: float = 5.0,
                 request_timeout: float = 10.0, connect=None):
        """
        Args:
            url: Home Assistant base URL (e.g. http://homeassistant.local:8123)
            token: Long-lived access token (HA profile -> Security)
            reconnect_delay: Seconds to wait before reconnecting
            request_timeout: Seconds to wait for a command result
            connect: Optional coroutine factory returning a websocket-like
                object (send_json / receive_json / close). Defaults to aiohttp;
                injectable for tests.
        """
        self.ws_url = websocket_url(url)
        self.token = token
        self.reconnect_delay = reconnect_delay
        self.request_timeout = request_timeout
        self._connect = connect or self._aiohttp_connect

        self.states: Dict[str, Dict[str, Any]] = {}
        self.connected = asyncio.Event()
        self._listeners: List[StateListener] = []
        self._ws = None
        self._session = None
        self._ids = itertools.count(1)
        self._pending: Dict[int, asyncio.Future] = {}
        self._task: Optional[asyncio.Task] = None

    # ---- lifecycle --------------------------------------------------------
    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        await self._close()

    @property
    def is_connected(self) -> bool:
        return self.connected.is_set()

    def add_listener(self, listener: StateListener):
        """Register an async callback for every entity state change."""
        self._listeners.append(listener)

    # ---- public API -------------------------------------------------------
    def get_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.states.get(entity_id)

    async def call_service(self, domain: str, service: str,
                           service_data: Optional[Dict[str, Any]] = None,
                           target: Optional[Dict[str, Any]] = None) -> bool:
        """Call a Home Assistant service. Returns True on success."""
        if not self.is_connected:
            logger.warning(f"Home Assistant not connected; dropping {domain}.{service}")
            return False
        message: Dict[str, Any] = {"type": "call_service", "domain": domain, "service": service}
        if service_data:
            message["service_data"] = service_data
        if target:
            message["target"] = target
        try:
            result = await self._request(message)
        except Exception as e:
            logger.error(f"Home Assistant {domain}.{service} failed: {e}")
            return False
        if not result.get("success"):
            logger.error(f"Home Assistant {domain}.{service} rejected: {result.get('error')}")
            return False
        return True

    # ---- connection loop --------------------------------------------------
    async def _run(self):
        while True:
            try:
                await self._session_once()
            except asyncio.CancelledError:
                raise
            except HomeAssistantAuthError as e:
                logger.error(f"Home Assistant authentication failed ({e}); "
                             "check HA_TOKEN. Not retrying.")
                return
            except Exception as e:
                logger.warning(f"Home Assistant connection lost: {e}")
            finally:
                self.connected.clear()
                self._fail_pending(ConnectionError("Home Assistant connection closed"))
                await self._close()
            await asyncio.sleep(self.reconnect_delay)

    async def _session_once(self):
        self._ws = await self._connect(self.ws_url)
        self._ids = itertools.count(1)

        # Handshake: auth_required -> auth -> auth_ok | auth_invalid
        hello = await self._ws.receive_json()
        if hello.get("type") != "auth_required":
            raise ConnectionError(f"unexpected handshake message: {hello.get('type')}")
        await self._ws.send_json({"type": "auth", "access_token": self.token})
        reply = await self._ws.receive_json()
        if reply.get("type") == "auth_invalid":
            raise HomeAssistantAuthError(reply.get("message", "auth_invalid"))
        if reply.get("type") != "auth_ok":
            raise ConnectionError(f"unexpected auth reply: {reply.get('type')}")
        logger.info(f"Connected to Home Assistant {reply.get('ha_version', '')}".strip())

        # Subscribe before fetching states so no change is missed in between.
        subscribe_id = next(self._ids)
        await self._ws.send_json({"id": subscribe_id, "type": "subscribe_events",
                                  "event_type": "state_changed"})
        states_id = next(self._ids)
        await self._ws.send_json({"id": states_id, "type": "get_states"})

        self.connected.set()
        while True:
            message = await self._ws.receive_json()
            await self._dispatch(message, states_id)

    async def _dispatch(self, message: Dict[str, Any], states_id: int):
        msg_type = message.get("type")
        if msg_type == "result":
            msg_id = message.get("id")
            if msg_id == states_id and message.get("success"):
                self.states = {s["entity_id"]: s for s in message.get("result") or []}
                logger.info(f"Home Assistant: {len(self.states)} entities loaded")
            future = self._pending.pop(msg_id, None)
            if future and not future.done():
                future.set_result(message)
        elif msg_type == "event":
            event = message.get("event", {})
            if event.get("event_type") == "state_changed":
                await self._on_state_changed(event.get("data", {}))

    async def _on_state_changed(self, data: Dict[str, Any]):
        entity_id = data.get("entity_id")
        if not entity_id:
            return
        old_state, new_state = data.get("old_state"), data.get("new_state")
        if new_state is None:
            self.states.pop(entity_id, None)  # entity removed
        else:
            self.states[entity_id] = new_state
        for listener in self._listeners:
            try:
                await listener(entity_id, old_state, new_state)
            except Exception as e:
                logger.error(f"Home Assistant listener error: {e}")

    async def _request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        msg_id = next(self._ids)
        future = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = future
        try:
            await self._ws.send_json({"id": msg_id, **message})
            return await asyncio.wait_for(future, timeout=self.request_timeout)
        finally:
            self._pending.pop(msg_id, None)

    def _fail_pending(self, error: Exception):
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    async def _close(self):
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        session, self._session = self._session, None
        if session is not None:
            await session.close()

    async def _aiohttp_connect(self, url: str):
        import aiohttp

        self._session = aiohttp.ClientSession()
        return await self._session.ws_connect(url, heartbeat=30, max_msg_size=0)


def build_home_assistant_client(config: Dict[str, Any]) -> Optional[HomeAssistantClient]:
    """Create a client from the `home_assistant` config section, or None if unset."""
    url, token = config.get("url"), config.get("token")
    if not (is_configured(url) and is_configured(token)):
        logger.info("Home Assistant not configured (set HA_URL and HA_TOKEN); integration disabled")
        return None
    return HomeAssistantClient(
        url, token,
        reconnect_delay=float(config.get("reconnect_delay", 5.0)),
        request_timeout=float(config.get("request_timeout", 10.0)),
    )
