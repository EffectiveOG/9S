from jarvis.components.automation.controllers.blinds_controller import BlindsController
from jarvis.components.automation.automation_component import AutomationComponent


async def test_blinds_virtual_device_behaviour():
    # No ip_address -> virtual device; execute() succeeds without a session.
    b = BlindsController({"id": "b", "type": "blinds"})
    assert await b.execute("close") is True
    assert (await b.get_state())["open"] is False
    assert await b.execute("open") is True
    assert (await b.get_state())["position"] == 100
    assert await b.execute("set_position", 50) is True
    assert (await b.get_state())["position"] == 50
    assert await b.execute("set_position", 999) is False   # out of range
    assert await b.execute("does_not_exist") is False


async def test_blinds_registered_in_automation():
    auto = AutomationComponent({
        "devices": [{"type": "blinds", "id": "living_room_blinds"}],
        "scenes_file": "config/scenes.json",
    })
    assert "living_room_blinds" in auto.controllers
    assert isinstance(auto.controllers["living_room_blinds"], BlindsController)
