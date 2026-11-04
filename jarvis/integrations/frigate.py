# jarvis/integrations/frigate.py
"""Frigate NVR integration over MQTT (Frigate 0.16+ / 0.17).

Frigate does the heavy lifting for the camera network (decoding, object
detection, face recognition, license plates, zones). Jarvis only consumes the
results it publishes on MQTT:

    <prefix>/available               "online" | "offline"
    <prefix>/events                  tracked object lifecycle (new/update/end)
    <prefix>/tracked_object_update   face / lpr / description enrichments
    <prefix>/<camera|zone>/<label>   number of <label> objects currently there

Reference: https://docs.frigate.video/integrations/mqtt
"""

import asyncio
import json
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# (topic, payload) -> None
MessageHandler = Callable[[str, bytes], Awaitable[None]]


def parse_sub_label(raw: Any) -> Tuple[Optional[str], Optional[float]]:
    """Frigate sends sub_label as ["Name", score], "Name" (older) or null."""
    if isinstance(raw, (list, tuple)) and raw:
        name = raw[0]
        score = raw[1] if len(raw) > 1 else None
        return (name or None), (float(score) if score is not None else None)
    if isinstance(raw, str) and raw:
        return raw, None
    return None, None


def normalize_tracked_object(obj: Dict[str, Any], api_url: Optional[str] = None) -> Dict[str, Any]:
    """Reduce a Frigate tracked-object payload to the fields Jarvis uses."""
    sub_label, sub_label_score = parse_sub_label(obj.get("sub_label"))
    plate = obj.get("recognized_license_plate")
    normalized = {
        "id": obj.get("id"),
        "camera": obj.get("camera"),
        "label": obj.get("label"),
        "sub_label": sub_label,
        "sub_label_score": sub_label_score,
        "score": obj.get("top_score", obj.get("score")),
        "zones": list(obj.get("current_zones") or []),
        "entered_zones": list(obj.get("entered_zones") or []),
        "plate": plate or None,
        "plate_score": obj.get("recognized_license_plate_score"),
        "start_time": obj.get("start_time"),
        "end_time": obj.get("end_time"),
        "stationary": bool(obj.get("stationary", False)),
        "false_positive": bool(obj.get("false_positive", False)),
        "has_snapshot": bool(obj.get("has_snapshot", False)),
    }
    if api_url and normalized["id"] and normalized["has_snapshot"]:
        normalized["snapshot_url"] = f"{api_url.rstrip('/')}/api/events/{normalized['id']}/snapshot.jpg"
    return normalized


class FrigateMQTTClient:
    """Subscribes to Frigate's MQTT topics and forwards messages to a handler.

    Reconnects on its own. Uses aiomqtt (imported lazily so the rest of Jarvis
    works without it).
    """

    def __init__(self, host: str, handler: MessageHandler, port: int = 1883,
                 username: Optional[str] = None, password: Optional[str] = None,
                 topic_prefix: str = "frigate", client_id: str = "jarvis",
                 reconnect_delay: float = 5.0):
        self.host = host
        self.port = port
        self.username = username or None
        self.password = password or None
        self.topic_prefix = topic_prefix.rstrip("/")
        self.client_id = client_id
        self.reconnect_delay = reconnect_delay
        self.handler = handler
        self.connected = False
        self._task: Optional[asyncio.Task] = None

    @property
    def topics(self):
        p = self.topic_prefix
        return [f"{p}/available", f"{p}/events", f"{p}/tracked_object_update", f"{p}/+/+"]

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
        self.connected = False

    async def _run(self):
        try:
            import aiomqtt
        except ImportError:
            logger.error("aiomqtt is not installed (pip install aiomqtt); Frigate integration disabled")
            return

        while True:
            try:
                async with aiomqtt.Client(
                    hostname=self.host, port=self.port,
                    username=self.username, password=self.password,
                    identifier=self.client_id,
                ) as client:
                    for topic in self.topics:
                        await client.subscribe(topic)
                    self.connected = True
                    logger.info(f"Connected to MQTT broker {self.host}:{self.port} "
                                f"(Frigate prefix '{self.topic_prefix}')")
                    async for message in client.messages:
                        try:
                            await self.handler(message.topic.value, bytes(message.payload or b""))
                        except Exception as e:
                            logger.error(f"Error handling Frigate message on {message.topic.value}: {e}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"MQTT connection lost ({e}); retrying in {self.reconnect_delay}s")
            finally:
                self.connected = False
            await asyncio.sleep(self.reconnect_delay)


def decode_json(payload: bytes) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None
