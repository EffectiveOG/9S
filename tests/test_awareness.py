import json

from jarvis.components.automation.automation_component import AutomationComponent
from jarvis.components.awareness.awareness_component import AwarenessComponent
from jarvis.components.memory.memory_component import MemoryComponent
from jarvis.core.message import Message
from jarvis.integrations.frigate import parse_sub_label


async def _awareness(**config):
    comp = AwarenessComponent({"frigate": {"api_url": "http://frigate:5000"}, **config})
    received = []

    class _Sink:
        async def put(self, message):
            received.append(message)

    comp.subscribers.add(_Sink())
    await comp.start()
    return comp, received


def _event(kind, obj_id="1700.0-abc", camera="entree", label="person", **after):
    payload = {"type": kind, "before": {}, "after": {
        "id": obj_id, "camera": camera, "label": label, "sub_label": None,
        "top_score": 0.9, "current_zones": [], "entered_zones": [], "has_snapshot": True,
        "false_positive": False, **after,
    }}
    return json.dumps(payload).encode()


def _types(received):
    return [m.message_type for m in received]


def test_parse_sub_label_formats():
    assert parse_sub_label(["Alice", 0.91]) == ("Alice", 0.91)
    assert parse_sub_label("Alice") == ("Alice", None)
    assert parse_sub_label(None) == (None, None)


async def test_mqtt_disabled_without_host():
    comp = AwarenessComponent({"frigate": {"mqtt": {"host": "${MQTT_HOST}"}}})
    assert comp.mqtt_client is None


async def test_face_identification_and_movement():
    comp, received = await _awareness()
    await comp.handle_frigate_message("frigate/available", b"online")
    await comp.handle_frigate_message("frigate/events", _event("new"))
    await comp.handle_frigate_message(
        "frigate/events", _event("update", sub_label=["Alice", 0.93], current_zones=["porche"]))
    # Same identity on the next update: no duplicate announcement
    await comp.handle_frigate_message(
        "frigate/events", _event("update", sub_label=["Alice", 0.95], current_zones=["porche"]))
    await comp.handle_frigate_message(
        "frigate/events", _event("update", sub_label=["Alice", 0.95], current_zones=["allee"]))
    await comp.handle_frigate_message("frigate/events", _event("end", sub_label=["Alice", 0.95]))

    assert _types(received) == ["object_detected", "person_identified", "person_moved", "object_ended"]
    identified = received[1].data
    assert identified["name"] == "Alice" and identified["camera"] == "entree"
    assert identified["snapshot_url"] == "http://frigate:5000/api/events/1700.0-abc/snapshot.jpg"
    assert received[2].data["previous_zones"] == ["porche"]
    assert comp.people["Alice"]["zones"] == ["allee"]
    assert comp.tracked == {}  # ended objects are forgotten
    assert comp.frigate_online is True


async def test_tracked_object_update_face_and_lpr():
    comp, received = await _awareness()
    await comp.handle_frigate_message("frigate/events", _event("new", obj_id="p1"))
    await comp.handle_frigate_message("frigate/tracked_object_update", json.dumps(
        {"type": "face", "id": "p1", "name": "Bob", "score": 0.88, "camera": "entree"}).encode())
    await comp.handle_frigate_message("frigate/tracked_object_update", json.dumps(
        {"type": "face", "id": "p1", "name": "unknown", "score": 0.4, "camera": "entree"}).encode())
    await comp.handle_frigate_message("frigate/events", _event("new", obj_id="c1", label="car"))
    await comp.handle_frigate_message("frigate/tracked_object_update", json.dumps(
        {"type": "lpr", "id": "c1", "name": "Voiture de Bob", "plate": "AB123CD",
         "score": 0.97, "camera": "allee"}).encode())
    # The plate also shows up on the event itself: still only one announcement
    await comp.handle_frigate_message("frigate/events", _event(
        "update", obj_id="c1", label="car", recognized_license_plate="AB123CD"))

    assert _types(received) == ["object_detected", "person_identified",
                                "object_detected", "vehicle_identified"]
    assert received[1].data["name"] == "Bob"
    assert received[3].data["plate"] == "AB123CD"
    assert received[3].data["name"] == "Voiture de Bob"
    assert "AB123CD" in comp.vehicles


async def test_area_counts_publish_transitions_only():
    comp, received = await _awareness()
    await comp.handle_frigate_message("frigate/salon/person", b"0")      # startup zero: silent
    await comp.handle_frigate_message("frigate/salon/person", b"1")
    await comp.handle_frigate_message("frigate/salon/person", b"1")      # unchanged: silent
    await comp.handle_frigate_message("frigate/salon/motion", b"ON")     # not a count
    await comp.handle_frigate_message("frigate/salon/person", b"0")

    assert [(m.data["count"], m.data["previous"]) for m in received] == [(1, 0), (0, 1)]
    assert comp.occupied_areas() == []


async def test_home_assistant_presence():
    comp, received = await _awareness()
    await comp.handle_ha_state_change(
        "person.alice", {"state": "not_home"},
        {"state": "home", "attributes": {"friendly_name": "Alice"}})
    await comp.handle_ha_state_change("light.salon", {"state": "off"}, {"state": "on"})  # filtered
    assert _types(received) == ["home_presence_update"]
    assert received[0].data == {"entity_id": "person.alice", "name": "Alice",
                                "state": "home", "previous": "not_home"}
    assert comp.snapshot()["home_presence"]["person.alice"]["state"] == "home"


async def test_area_rules_fire_on_threshold_crossing():
    auto = AutomationComponent({"devices": [], "scenes_file": "config/scenes.json"})
    auto.rules = {"occupancy_rules": [
        {"name": "empty", "area": "salon", "conditions": {"max_occupancy": 0},
         "actions": [{"device_id": "lights", "type": "power_off"}]},
        {"name": "crowd", "area": "salon", "conditions": {"min_occupancy": 2},
         "actions": [{"device_id": "tv", "type": "power_on"}]},
    ]}
    fired = []

    async def record(actions):
        fired.extend(a["type"] for a in actions)

    auto._execute_rule_actions = record

    async def presence(count, previous, area="salon"):
        await auto.handle_message(Message("awareness", "presence_update", {
            "area": area, "label": "person", "count": count, "previous": previous}))

    await presence(1, 0)
    await presence(2, 1)
    await presence(3, 2)        # still above threshold: no re-trigger
    await presence(0, 3)
    await presence(0, 1, area="cuisine")  # other area
    assert fired == ["power_on", "power_off"]


async def test_memory_keeps_awareness_history(tmp_path):
    mem = MemoryComponent({"db_path": str(tmp_path / "t.db")})
    await mem.start()
    try:
        await mem.process_message(Message("awareness", "person_identified",
                                          {"name": "Alice", "camera": "entree", "zones": []}))
        await mem.process_message(Message("awareness", "vehicle_identified",
                                          {"plate": "AB123CD", "camera": "allee"}))
        history = await mem.get_awareness_history()
        assert [h["subject"] for h in history] == ["AB123CD", "Alice"]
        alice = await mem.get_awareness_history("Alice")
        assert alice[0]["camera"] == "entree" and alice[0]["data"]["name"] == "Alice"
    finally:
        await mem.stop()
