# jarvis/components/awareness/awareness_component.py

import fnmatch
import time
from typing import Any, Dict, List, Optional

from ...core.base_component import BaseComponent
from ...core.message import Message
from ...integrations.frigate import FrigateMQTTClient, decode_json, normalize_tracked_object
from ...integrations.home_assistant import is_configured
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)

# Frigate "names" that mean no identity was recognized.
_UNKNOWN_NAMES = {None, "", "unknown", "none"}


class AwarenessComponent(BaseComponent):
    """Situational awareness of the house: who/what is where.

    Fuses Frigate (camera network: tracked objects, faces, plates, per-zone
    counts) and Home Assistant (person / presence entities) into one world
    model, and publishes the meaningful changes on the message bus:

        presence_update        an area's object count changed
        object_detected        Frigate started tracking an object
        object_ended           Frigate stopped tracking an object
        person_identified      a face was recognized (once per tracked object)
        person_moved           an identified person changed camera/zones
        vehicle_identified     a license plate was read (once per tracked object)
        home_presence_update   a Home Assistant person/presence entity changed
    """

    def __init__(self, config: Dict[str, Any], ha_client=None, mqtt_client=None):
        """
        Args:
            config: Configuration dictionary containing:
                - frigate: {mqtt: {host, port, username, password}, topic_prefix, api_url}
                - ha_entities: fnmatch patterns of HA entities to follow (default ["person.*"])
                - max_tracked: cap on concurrently tracked objects kept in memory
            ha_client: Optional HomeAssistantClient for presence entities
            mqtt_client: Optional pre-built Frigate client (tests)
        """
        super().__init__("awareness")
        self.config = config
        frigate = config.get("frigate", {})
        self.topic_prefix = frigate.get("topic_prefix", "frigate").rstrip("/")
        api_url = frigate.get("api_url")
        self.api_url = api_url if is_configured(api_url) else None
        self.ha_entities: List[str] = config.get("ha_entities", ["person.*"])
        self.max_tracked = int(config.get("max_tracked", 500))
        self.ha_client = ha_client
        self._ha_listening = False

        self.mqtt_client = mqtt_client or self._build_mqtt_client(frigate)

        # World model
        self.frigate_online: Optional[bool] = None
        self.areas: Dict[str, Dict[str, int]] = {}        # area -> label -> count
        self.tracked: Dict[str, Dict[str, Any]] = {}      # Frigate id -> object
        self.people: Dict[str, Dict[str, Any]] = {}       # name -> last sighting
        self.vehicles: Dict[str, Dict[str, Any]] = {}     # plate -> last sighting
        self.home_presence: Dict[str, Dict[str, Any]] = {}  # HA entity -> state

    def _build_mqtt_client(self, frigate: Dict[str, Any]) -> Optional[FrigateMQTTClient]:
        mqtt = frigate.get("mqtt", {})
        host = mqtt.get("host")
        if not is_configured(host):
            logger.info("Frigate MQTT not configured (set MQTT_HOST); camera awareness disabled")
            return None
        username, password = mqtt.get("username"), mqtt.get("password")
        return FrigateMQTTClient(
            host=host,
            handler=self.handle_frigate_message,
            port=int(mqtt.get("port", 1883)),
            username=username if is_configured(username) else None,
            password=password if is_configured(password) else None,
            topic_prefix=self.topic_prefix,
            client_id=mqtt.get("client_id", "jarvis"),
        )

    # ---- lifecycle --------------------------------------------------------
    async def start(self):
        await super().start()
        if self.mqtt_client:
            await self.mqtt_client.start()
        if self.ha_client and not self._ha_listening:
            self.ha_client.add_listener(self.handle_ha_state_change)
            self._ha_listening = True  # survives stop/start (component recovery)
        logger.info("Awareness component started")

    async def stop(self):
        if self.mqtt_client:
            await self.mqtt_client.stop()
        await super().stop()

    # ---- queries ----------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Current world model (served by /api/awareness)."""
        return {
            "frigate_online": self.frigate_online,
            "mqtt_connected": bool(self.mqtt_client and self.mqtt_client.connected),
            "home_assistant_connected": bool(self.ha_client and self.ha_client.is_connected),
            "areas": self.areas,
            "active_objects": list(self.tracked.values()),
            "people": self.people,
            "vehicles": self.vehicles,
            "home_presence": self.home_presence,
        }

    def occupied_areas(self, label: str = "person") -> List[str]:
        return [area for area, counts in self.areas.items() if counts.get(label, 0) > 0]

    # ---- Frigate ----------------------------------------------------------
    async def handle_frigate_message(self, topic: str, payload: bytes):
        prefix = self.topic_prefix + "/"
        if not topic.startswith(prefix):
            return
        parts = topic[len(prefix):].split("/")

        if parts == ["available"]:
            self.frigate_online = payload.decode(errors="ignore").strip() == "online"
            logger.info(f"Frigate is {'online' if self.frigate_online else 'offline'}")
        elif parts == ["events"]:
            data = decode_json(payload)
            if data:
                await self._on_event(data)
        elif parts == ["tracked_object_update"]:
            data = decode_json(payload)
            if data:
                await self._on_tracked_object_update(data)
        elif len(parts) == 2:
            await self._on_count(parts[0], parts[1], payload)
        self._refresh_state()

    async def _on_count(self, area: str, label: str, payload: bytes):
        try:
            count = int(payload.decode().strip())
        except (ValueError, UnicodeDecodeError):
            return  # motion ON/OFF, review_status, ... — not an object count
        previous = self.areas.get(area, {}).get(label)
        if previous == count:
            return
        self.areas.setdefault(area, {})[label] = count
        # Frigate announces zeros at startup; that is not a transition.
        if previous is None and count == 0:
            return
        await self._emit("presence_update", {
            "area": area, "label": label, "count": count, "previous": previous,
        })

    async def _on_event(self, data: Dict[str, Any]):
        event_type = data.get("type")
        after = data.get("after") or {}
        if not after.get("id"):
            return
        obj = normalize_tracked_object(after, self.api_url)
        if obj["false_positive"]:
            return
        previous = self.tracked.get(obj["id"])
        # Keep enrichments that arrived via tracked_object_update.
        if previous:
            for key in ("identified_as", "plate_reported", "description"):
                if key in previous:
                    obj[key] = previous[key]

        if event_type == "end":
            self.tracked.pop(obj["id"], None)
        else:
            self._remember(obj)

        if event_type == "new":
            await self._emit("object_detected", self._public(obj))

        if obj["label"] == "person":
            name = obj["sub_label"] or obj.get("identified_as")
            if name and obj.get("identified_as") != name:
                await self._identify_person(obj, name, obj["sub_label_score"])
            elif name and previous and event_type != "end" and (
                    (previous.get("camera"), previous.get("zones")) != (obj["camera"], obj["zones"])):
                await self._person_seen(obj, name, moved=True)

        if obj["plate"]:
            await self._identify_vehicle(obj, obj["plate"], obj["plate_score"], obj["sub_label"])

        if event_type == "end":
            await self._emit("object_ended", self._public(obj))

    async def _on_tracked_object_update(self, data: Dict[str, Any]):
        obj_id = data.get("id")
        obj = self.tracked.get(obj_id) or {
            "id": obj_id, "camera": data.get("camera"), "zones": [], "label": None,
        }
        update_type = data.get("type")
        if update_type == "face":
            obj.setdefault("label", "person")
            await self._identify_person(obj, data.get("name"), data.get("score"))
        elif update_type == "lpr":
            await self._identify_vehicle(obj, data.get("plate"), data.get("score"), data.get("name"))
        elif update_type == "description" and obj_id in self.tracked:
            self.tracked[obj_id]["description"] = data.get("description")

    async def _identify_person(self, obj: Dict[str, Any], name: Optional[str], score: Optional[float]):
        if str(name).lower() in _UNKNOWN_NAMES:
            return
        first_time = obj.get("identified_as") != name
        obj["identified_as"] = name
        if obj.get("id") in self.tracked:
            self.tracked[obj["id"]]["identified_as"] = name
        await self._person_seen(obj, name, moved=False, score=score, announce=first_time)

    async def _person_seen(self, obj: Dict[str, Any], name: str, moved: bool,
                           score: Optional[float] = None, announce: bool = False):
        previous = self.people.get(name)
        sighting = {
            "name": name,
            "camera": obj.get("camera"),
            "zones": obj.get("zones", []),
            "event_id": obj.get("id"),
            "score": score if score is not None else (previous or {}).get("score"),
            "last_seen": time.time(),
        }
        self.people[name] = sighting
        if announce:
            await self._emit("person_identified", {**sighting, "snapshot_url": obj.get("snapshot_url")})
        elif moved:
            await self._emit("person_moved", {
                **sighting,
                "previous_camera": (previous or {}).get("camera"),
                "previous_zones": (previous or {}).get("zones", []),
            })

    async def _identify_vehicle(self, obj: Dict[str, Any], plate: Optional[str],
                                score: Optional[float], name: Optional[str]):
        if not plate:
            return
        sighting = {
            "plate": plate,
            "name": None if str(name).lower() in _UNKNOWN_NAMES else name,
            "camera": obj.get("camera"),
            "zones": obj.get("zones", []),
            "event_id": obj.get("id"),
            "score": score,
            "last_seen": time.time(),
        }
        self.vehicles[plate] = sighting
        if obj.get("plate_reported") == plate:
            return
        obj["plate_reported"] = plate
        if obj.get("id") in self.tracked:
            self.tracked[obj["id"]]["plate_reported"] = plate
        await self._emit("vehicle_identified", {**sighting, "snapshot_url": obj.get("snapshot_url")})

    def _remember(self, obj: Dict[str, Any]):
        self.tracked[obj["id"]] = obj
        # Bound memory if Frigate "end" events are ever missed.
        while len(self.tracked) > self.max_tracked:
            self.tracked.pop(next(iter(self.tracked)))

    @staticmethod
    def _public(obj: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in obj.items() if k not in ("plate_reported",)}

    # ---- Home Assistant ---------------------------------------------------
    async def handle_ha_state_change(self, entity_id: str, old_state: Optional[Dict[str, Any]],
                                     new_state: Optional[Dict[str, Any]]):
        if not any(fnmatch.fnmatch(entity_id, pattern) for pattern in self.ha_entities):
            return
        old = (old_state or {}).get("state")
        new = (new_state or {}).get("state")
        if new_state is None:
            self.home_presence.pop(entity_id, None)
        else:
            self.home_presence[entity_id] = {
                "state": new,
                "name": new_state.get("attributes", {}).get("friendly_name", entity_id),
                "last_changed": new_state.get("last_changed"),
            }
        if old != new:
            await self._emit("home_presence_update", {
                "entity_id": entity_id,
                "name": (self.home_presence.get(entity_id) or {}).get("name", entity_id),
                "state": new,
                "previous": old,
            })
        self._refresh_state()

    # ---- helpers ----------------------------------------------------------
    async def _emit(self, message_type: str, data: Dict[str, Any]):
        logger.debug(f"awareness: {message_type} {data}")
        await self.publish(Message(sender=self.name, message_type=message_type, data=data))

    def _refresh_state(self):
        self.update_state({
            "frigate_online": self.frigate_online,
            "occupied_areas": self.occupied_areas(),
            "people_known": sorted(self.people),
            "active_objects": len(self.tracked),
        })
