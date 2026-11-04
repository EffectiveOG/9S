# jarvis/components/automation/controllers/home_assistant_controller.py

from typing import Any, Dict, Optional, Tuple
from .base_controller import BaseController
from ....utils.logging_utils import get_logger

logger = get_logger(__name__)

# Domains whose on/off services have different names.
_ON_OFF_SERVICES = {
    "cover": ("open_cover", "close_cover"),
    "scene": ("turn_on", None),
    "script": ("turn_on", "turn_off"),
    "button": ("press", None),
}

# Security-sensitive domains: only explicitly configured `actions` overrides
# are allowed, never the generic power_on/toggle mapping (a stray gesture or
# misheard phrase must not unlock a door or disarm the alarm).
_SENSITIVE_DOMAINS = {"lock", "alarm_control_panel"}


class HomeAssistantController(BaseController):
    """Drives one Home Assistant entity using Jarvis' device action vocabulary.

    Scenes, speech and gestures keep emitting the same actions they always did
    (power_on, set_brightness, set_position, set_input, ...); this controller
    translates them to Home Assistant service calls based on the entity domain.

    Config:
        id:        Jarvis device id (as referenced in scenes.json)
        type:      "home_assistant"
        entity_id: Home Assistant entity (light.salon, cover.salon, media_player.tv)
        scenes:    Optional {value: "scene.xxx"} map for the set_scene action
        inputs:    Optional {value: "HDMI 1"} map for the set_input action
        actions:   Optional per-action overrides:
                   {"set_game_mode": {"service": "switch.turn_on",
                                      "target": {"entity_id": "switch.tv_game_mode"},
                                      "data": {}, "value_key": null}}
    """

    def __init__(self, config: Dict[str, Any], client):
        super().__init__(config)
        self.client = client
        self.entity_id = config["entity_id"]
        self.domain = self.entity_id.split(".", 1)[0]
        self.scenes = config.get("scenes", {})
        self.inputs = config.get("inputs", {})
        self.overrides = config.get("actions", {})

    async def execute(self, action: str, value: Any = None) -> bool:
        call = self._resolve(action, value)
        if call is None:
            logger.warning(f"{self.device_id}: action '{action}' not supported for {self.entity_id}")
            return False
        service, data, target = call
        domain, _, service_name = service.partition(".")
        return await self.client.call_service(domain, service_name, data or None, target)

    async def get_state(self) -> Dict[str, Any]:
        state = self.client.get_state(self.entity_id)
        if state is None:
            return {"entity_id": self.entity_id, "available": False}
        return {
            "entity_id": self.entity_id,
            "state": state.get("state"),
            "available": state.get("state") != "unavailable",
            "attributes": state.get("attributes", {}),
        }

    # ---- action translation ----------------------------------------------
    def _resolve(self, action: str, value: Any) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        """Return (domain.service, service_data, target) or None if unsupported."""
        target = {"entity_id": self.entity_id}

        if action in self.overrides:
            spec = self.overrides[action]
            data = dict(spec.get("data", {}))
            if spec.get("value_key") and value is not None:
                data[spec["value_key"]] = value
            return spec["service"], data, spec.get("target", target)

        if self.domain in _SENSITIVE_DOMAINS:
            return None

        if action == "call_service" and isinstance(value, dict):
            return value["service"], value.get("data", {}), value.get("target", target)

        on_service, off_service = _ON_OFF_SERVICES.get(self.domain, ("turn_on", "turn_off"))
        if action in ("power_on", "turn_on", "open"):
            return f"{self.domain}.{on_service}", {}, target
        if action in ("power_off", "turn_off", "close") and off_service:
            return f"{self.domain}.{off_service}", {}, target
        if action in ("power_toggle", "toggle") and self.domain not in ("scene", "button"):
            return f"{self.domain}.toggle", {}, target

        if action == "set_scene" and value is not None:
            scene = self.scenes.get(value) or (value if "." in str(value) else None)
            if scene is None:
                return None
            return "scene.turn_on", {}, {"entity_id": scene}

        if self.domain == "light":
            if action == "set_brightness" and isinstance(value, (int, float)):
                return "light.turn_on", {"brightness_pct": max(0, min(100, int(value)))}, target
            if action == "set_color" and isinstance(value, (list, tuple)) and len(value) == 3:
                return "light.turn_on", {"rgb_color": list(value)}, target
            if action == "set_color_temp" and isinstance(value, (int, float)):
                return "light.turn_on", {"color_temp_kelvin": int(value)}, target
            if action in ("increase", "decrease"):
                step = 10 if action == "increase" else -10
                return "light.turn_on", {"brightness_step_pct": step}, target

        if self.domain == "cover" and action == "set_position" and isinstance(value, (int, float)):
            return "cover.set_cover_position", {"position": max(0, min(100, int(value)))}, target

        if self.domain == "media_player":
            if action == "set_input" and value is not None:
                return "media_player.select_source", {"source": self.inputs.get(value, value)}, target
            if action == "set_volume" and isinstance(value, (int, float)):
                return "media_player.volume_set", {"volume_level": max(0, min(100, value)) / 100}, target
            if action == "increase":
                return "media_player.volume_up", {}, target
            if action == "decrease":
                return "media_player.volume_down", {}, target
            if action == "mute":
                return "media_player.volume_mute", {"is_volume_muted": bool(value if value is not None else True)}, target

        if self.domain == "climate" and action == "set_temperature" and isinstance(value, (int, float)):
            return "climate.set_temperature", {"temperature": value}, target

        return None
