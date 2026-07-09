from jarvis.components.memory.memory_component import MemoryComponent
from jarvis.core.message import Message


async def test_store_and_retrieve(tmp_path):
    mem = MemoryComponent({"db_path": str(tmp_path / "t.db"), "retention_days": 30})
    await mem.start()
    try:
        await mem.process_message(Message("vision", "vision_update", {
            "objects": [{"class": "person", "confidence": 0.9, "bbox": [1, 2, 3, 4], "track_id": 7}],
            "faces": [{"identity": "x", "confidence": 0.5, "location": [1, 2, 3, 4]}],
            "gestures": [{"gesture": "wave", "confidence": 0.6, "hand_index": 0}],
        }))
        await mem.process_message(Message("audio", "speech_recognized", {"text": "hi"}))
        await mem.process_message(Message("auto", "device_update",
                                          {"device_id": "tv", "state": {"on": True}}))

        objs = await mem.get_recent_detections("object", 5)
        assert len(objs) == 1
        assert objs[0]["bbox"] == [1, 2, 3, 4]  # JSON field parsed back to a list

        stats = await mem.get_statistics()
        assert stats["object_detections_count"] == 1
        assert stats["face_detections_count"] == 1
        assert stats["gesture_detections_count"] == 1

        speech = mem.conn.execute("SELECT count(*) FROM speech_events").fetchone()[0]
        devices = mem.conn.execute("SELECT count(*) FROM device_events").fetchone()[0]
        assert speech == 1 and devices == 1
    finally:
        await mem.stop()


async def test_unknown_detection_type_returns_empty(tmp_path):
    mem = MemoryComponent({"db_path": str(tmp_path / "t.db")})
    await mem.start()
    try:
        assert await mem.get_recent_detections("bogus") == []
    finally:
        await mem.stop()
