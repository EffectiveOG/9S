import asyncio

from jarvis.components.automation.automation_component import AutomationComponent
from jarvis.components.automation.controllers.home_assistant_controller import HomeAssistantController
from jarvis.integrations.home_assistant import (
    HomeAssistantClient, build_home_assistant_client, is_configured, websocket_url,
)


class FakeWebSocket:
    """Scripted Home Assistant server: answers auth, get_states and call_service."""

    def __init__(self, token="good", states=None):
        self.token = token
        self.states = states or []
        self.sent = []
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.inbox.put_nowait({"type": "auth_required", "ha_version": "2026.10.0"})

    async def send_json(self, message):
        self.sent.append(message)
        if message["type"] == "auth":
            ok = message["access_token"] == self.token
            await self.inbox.put({"type": "auth_ok" if ok else "auth_invalid"})
        elif message["type"] == "subscribe_events":
            await self.inbox.put({"id": message["id"], "type": "result", "success": True})
        elif message["type"] == "get_states":
            await self.inbox.put({"id": message["id"], "type": "result", "success": True,
                                  "result": self.states})
        elif message["type"] == "call_service":
            await self.inbox.put({"id": message["id"], "type": "result", "success": True,
                                  "result": {"context": {}}})

    async def receive_json(self):
        return await self.inbox.get()

    async def close(self):
        pass

    def push_state_change(self, entity_id, old, new):
        self.inbox.put_nowait({"id": 1, "type": "event", "event": {
            "event_type": "state_changed",
            "data": {"entity_id": entity_id,
                     "old_state": {"entity_id": entity_id, "state": old},
                     "new_state": {"entity_id": entity_id, "state": new}},
        }})


async def _connected_client(ws, token="good"):
    async def connect(url):
        return ws

    client = HomeAssistantClient("http://ha.local:8123", token, connect=connect, reconnect_delay=0.01)
    await client.start()
    await asyncio.wait_for(client.connected.wait(), timeout=2)
    await asyncio.sleep(0.05)  # let get_states result land
    return client


def test_config_helpers():
    assert websocket_url("http://ha.local:8123/") == "ws://ha.local:8123/api/websocket"
    assert websocket_url("https://ha.example.com") == "wss://ha.example.com/api/websocket"
    assert not is_configured("${HA_TOKEN}") and not is_configured("") and not is_configured(None)
    assert build_home_assistant_client({"url": "http://ha:8123", "token": "${HA_TOKEN}"}) is None
    assert build_home_assistant_client({"url": "http://ha:8123", "token": "abc"}) is not None


async def test_auth_states_events_and_service_call():
    ws = FakeWebSocket(states=[{"entity_id": "light.salon", "state": "off"}])
    client = await _connected_client(ws)
    try:
        assert ws.sent[0] == {"type": "auth", "access_token": "good"}
        assert client.get_state("light.salon")["state"] == "off"

        changes = []

        async def listener(entity_id, old, new):
            changes.append((entity_id, old["state"], new["state"]))

        client.add_listener(listener)
        ws.push_state_change("person.alice", "not_home", "home")
        await asyncio.sleep(0.05)
        assert changes == [("person.alice", "not_home", "home")]
        assert client.get_state("person.alice")["state"] == "home"

        ok = await client.call_service("light", "turn_on", {"brightness_pct": 40},
                                       {"entity_id": "light.salon"})
        assert ok is True
        call = ws.sent[-1]
        assert call["type"] == "call_service" and call["domain"] == "light"
        assert call["service_data"] == {"brightness_pct": 40}
        assert call["target"] == {"entity_id": "light.salon"}
    finally:
        await client.stop()


async def test_invalid_token_stops_without_retrying():
    attempts = []

    async def connect(url):
        attempts.append(url)
        return FakeWebSocket(token="good")

    client = HomeAssistantClient("http://ha.local:8123", "bad", connect=connect, reconnect_delay=0.01)
    await client.start()
    await asyncio.sleep(0.1)
    assert not client.is_connected
    assert len(attempts) == 1
    assert await client.call_service("light", "turn_on") is False
    await client.stop()


class RecordingClient:
    def __init__(self, states=None):
        self.calls = []
        self.states = states or {}

    async def call_service(self, domain, service, data=None, target=None):
        self.calls.append((f"{domain}.{service}", data, target))
        return True

    def get_state(self, entity_id):
        return self.states.get(entity_id)


async def test_controller_translates_jarvis_actions():
    client = RecordingClient()
    light = HomeAssistantController({"id": "lights", "type": "home_assistant", "entity_id": "light.salon",
                                     "scenes": {"movie": "scene.salon_cinema"}}, client)
    cover = HomeAssistantController({"id": "blinds", "type": "home_assistant", "entity_id": "cover.salon"}, client)
    tv = HomeAssistantController({"id": "tv", "type": "home_assistant", "entity_id": "media_player.tv",
                                  "inputs": {"hdmi1": "HDMI 1"},
                                  "actions": {"set_game_mode": {"service": "switch.turn_on",
                                                                "target": {"entity_id": "switch.tv_game"}}}},
                                 client)

    assert await light.execute("power_on")
    assert await light.execute("set_brightness", 150)
    assert await light.execute("set_scene", "movie")
    assert await cover.execute("set_position", 0)
    assert await cover.execute("power_off")
    assert await tv.execute("set_input", "hdmi1")
    assert await tv.execute("set_game_mode", True)
    assert not await light.execute("set_picture_mode", "game")  # unsupported -> False

    salon = {"entity_id": "light.salon"}
    assert client.calls == [
        ("light.turn_on", None, salon),
        ("light.turn_on", {"brightness_pct": 100}, salon),
        ("scene.turn_on", None, {"entity_id": "scene.salon_cinema"}),
        ("cover.set_cover_position", {"position": 0}, {"entity_id": "cover.salon"}),
        ("cover.close_cover", None, {"entity_id": "cover.salon"}),
        ("media_player.select_source", {"source": "HDMI 1"}, {"entity_id": "media_player.tv"}),
        ("switch.turn_on", None, {"entity_id": "switch.tv_game"}),
    ]


async def test_locks_require_explicit_override():
    client = RecordingClient()
    door = HomeAssistantController({"id": "door", "type": "home_assistant", "entity_id": "lock.front"}, client)
    assert not await door.execute("power_on")
    assert not await door.execute("toggle")
    assert not await door.execute("call_service", {"service": "lock.unlock"})
    assert client.calls == []


async def test_controller_state_from_cache():
    client = RecordingClient({"light.salon": {"state": "on", "attributes": {"brightness": 200}}})
    light = HomeAssistantController({"id": "l", "type": "home_assistant", "entity_id": "light.salon"}, client)
    assert (await light.get_state())["state"] == "on"
    missing = HomeAssistantController({"id": "m", "type": "home_assistant", "entity_id": "light.nope"}, client)
    assert (await missing.get_state())["available"] is False


def test_automation_builds_home_assistant_devices_only():
    devices = [
        {"type": "home_assistant", "id": "living_room_blinds", "entity_id": "cover.salon"},
        {"type": "tv", "id": "old_tv", "ip_address": "192.168.1.100"},  # legacy type: ignored
    ]
    auto = AutomationComponent({"devices": devices, "scenes_file": "config/scenes.json"},
                               ha_client=RecordingClient())
    assert list(auto.controllers) == ["living_room_blinds"]
    assert isinstance(auto.controllers["living_room_blinds"], HomeAssistantController)

    # Without Home Assistant no device is available.
    without_ha = AutomationComponent({"devices": devices, "scenes_file": "config/scenes.json"})
    assert without_ha.controllers == {}


async def test_shipped_config_drives_every_scene_device():
    import json

    with open("config/jarvis_config.json") as f:
        automation = json.load(f)["automation"]
    auto = AutomationComponent(automation, ha_client=RecordingClient())
    assert all(isinstance(c, HomeAssistantController) for c in auto.controllers.values())
    scene_devices = {d for scene in auto.scene_manager.scenes.values() for d in scene["devices"]}
    assert scene_devices <= set(auto.controllers)
