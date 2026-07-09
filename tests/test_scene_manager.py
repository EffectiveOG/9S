import json

from jarvis.components.automation.controllers.scene_manager import SceneManager


class FakeController:
    def __init__(self):
        self.calls = []

    async def execute(self, action, value=None):
        self.calls.append((action, value))
        return True


def _write_scenes(tmp_path):
    scenes = {
        "movie": {
            "devices": {"tv": [{"type": "power_on"}]},
            "exit_actions": {"tv": [{"type": "power_off"}]},
        }
    }
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps(scenes))
    return str(path)


async def test_activate_and_deactivate(tmp_path):
    ctrl = FakeController()
    sm = SceneManager({"scenes_file": _write_scenes(tmp_path), "controllers": {"tv": ctrl}})

    assert "movie" in sm.scenes
    assert await sm.activate_scene("movie") is True
    assert ("power_on", None) in ctrl.calls
    assert sm.get_active_scene() == "movie"

    assert await sm.deactivate_scene() is True
    assert ("power_off", None) in ctrl.calls
    assert sm.get_active_scene() is None


async def test_unknown_scene_returns_false(tmp_path):
    sm = SceneManager({"scenes_file": _write_scenes(tmp_path), "controllers": {}})
    assert await sm.activate_scene("does_not_exist") is False
