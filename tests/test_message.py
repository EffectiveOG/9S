from jarvis.core.message import Message


def test_roundtrip_to_from_dict():
    m = Message(sender="vision", message_type="vision_update", data={"a": 1})
    m2 = Message.from_dict(m.to_dict())
    assert m2.sender == "vision"
    assert m2.message_type == "vision_update"
    assert m2.data == {"a": 1}


def test_id_and_timestamp_autoset():
    m = Message("audio", "speech_recognized", {})
    assert m.id
    assert m.timestamp is not None


def test_type_predicates():
    assert Message("s", "command_do", {}).is_command()
    assert Message("s", "event_x", {}).is_event()
    assert Message("s", "response_y", {}).is_response()
    assert not Message("s", "vision_update", {}).is_command()


def test_create_response_links_correlation_id():
    m = Message("s", "command_do", {}, id="abc123")
    r = m.create_response({"ok": True})
    assert r.correlation_id == "abc123"
    assert r.data == {"ok": True}
    assert r.message_type == "response_command_do"
