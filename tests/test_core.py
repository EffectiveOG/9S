import asyncio

from jarvis.core.jarvis_core import JarvisCore
from jarvis.core.base_component import BaseComponent
from jarvis.core.message import Message


class _Producer(BaseComponent):
    async def start(self):
        await super().start()

    async def stop(self):
        await super().stop()


class _Consumer(BaseComponent):
    def __init__(self, name):
        super().__init__(name)
        self.received = []

    async def start(self):
        await super().start()

    async def stop(self):
        await super().stop()

    async def process_message(self, message):
        self.received.append(message)

    async def handle_message(self, message):
        self.received.append(message)


async def test_start_is_nonblocking_and_bus_delivers():
    core = JarvisCore("config/does_not_exist.json")
    producer = _Producer("vision")
    memory = _Consumer("memory")
    automation = _Consumer("automation")

    async def fake_init():
        core.components.update({"vision": producer, "memory": memory, "automation": automation})
        for name, comp in core.components.items():
            await comp.start()
            core.initialized_components.add(name)
        core._wire_message_bus()

    core._init_components = fake_init

    # start() must return promptly (previously it deadlocked on an infinite gather)
    await asyncio.wait_for(core.start(), timeout=5)
    assert core.is_running

    await producer.publish(Message("vision", "vision_update", {"objects": []}))
    await asyncio.sleep(0.2)
    assert memory.received, "memory did not receive the bus message"
    assert automation.received, "automation did not receive the bus message"

    await core.stop()
    assert core._loop_tasks == []
    assert core._pump_tasks == []


def test_expand_env_vars(monkeypatch):
    monkeypatch.setenv("FOO", "bar")
    out = JarvisCore._expand_env_vars({"a": "${FOO}", "b": ["${FOO}", 1], "c": 2})
    assert out == {"a": "bar", "b": ["bar", 1], "c": 2}


def test_parse_speech_command():
    core = JarvisCore.__new__(JarvisCore)  # skip __init__ (no event loop needed)

    class _SM:
        scenes = {"movie_night": {}, "gaming_intense": {}}

    class _Auto:
        scene_manager = _SM()

    core.components = {"automation": _Auto()}

    assert core._parse_speech_command("please play movie night") == {
        "type": "scene_control", "data": {"action": "activate", "scene": "movie_night"}
    }
    assert core._parse_speech_command("jarvis restart now") == {
        "type": "system_control", "data": {"action": "restart"}
    }
    assert core._parse_speech_command("what's the weather") is None
