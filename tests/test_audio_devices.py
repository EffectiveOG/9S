from jarvis.components.audio.audio_component import list_audio_devices, resolve_audio_device


class FakePyAudio:
    devices = [
        {"name": "MacBook Pro Microphone", "maxInputChannels": 1, "maxOutputChannels": 0},
        {"name": "MacBook Pro Speakers", "maxInputChannels": 0, "maxOutputChannels": 2},
        {"name": "AirPods Pro", "maxInputChannels": 1, "maxOutputChannels": 2},
        {"name": "Blue Yeti", "maxInputChannels": 2, "maxOutputChannels": 0},
    ]

    def get_device_count(self):
        return len(self.devices)

    def get_device_info_by_index(self, index):
        return self.devices[index]


def test_list_devices_by_kind():
    audio = FakePyAudio()
    assert list_audio_devices(audio, "input") == [
        (0, "MacBook Pro Microphone"), (2, "AirPods Pro"), (3, "Blue Yeti")]
    assert list_audio_devices(audio, "output") == [(1, "MacBook Pro Speakers"), (2, "AirPods Pro")]


def test_resolve_device():
    audio = FakePyAudio()
    # unset -> system default
    assert resolve_audio_device(audio, None) is None
    assert resolve_audio_device(audio, "") is None
    assert resolve_audio_device(audio, "${AUDIO_INPUT_DEVICE}") is None
    # index, as int or string from .env
    assert resolve_audio_device(audio, 3) == 3
    assert resolve_audio_device(audio, "3") == 3
    # part of the name, case-insensitive, restricted to the right kind
    assert resolve_audio_device(audio, "yeti") == 3
    assert resolve_audio_device(audio, "macbook", "input") == 0
    assert resolve_audio_device(audio, "macbook", "output") == 1
    # unknown name -> default (with a warning)
    assert resolve_audio_device(audio, "Rode NT-USB") is None
