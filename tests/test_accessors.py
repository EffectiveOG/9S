"""GUI support accessors added to the vision/audio components."""

from jarvis.components.vision.vision_component import VisionComponent
from jarvis.components.audio.audio_component import AudioComponent


def test_vision_get_frame():
    v = VisionComponent.__new__(VisionComponent)  # skip heavy __init__
    v.latest_frame = None
    assert v.get_frame() is None
    v.latest_frame = "FRAME"
    assert v.get_frame() == "FRAME"


def test_audio_process_audio_returns_and_clears():
    a = AudioComponent.__new__(AudioComponent)  # skip heavy __init__
    a._last_transcript = "hello jarvis"
    assert a.process_audio() == "hello jarvis"
    assert a.process_audio() is None  # cleared after read
