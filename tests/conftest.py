"""Shared pytest setup.

Stubs heavy/optional third-party libraries (torch, opencv, mediapipe, ...) that
aren't needed to test the pure-Python logic, so the test suite runs in a slim
CI environment without installing gigabytes of ML dependencies. A library is
only stubbed if it is *not* already installed, so real ones are used when
available.
"""

import importlib.abc
import importlib.machinery
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Make the project root importable regardless of where pytest is invoked.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Old tests targeting a previous API / real hardware — skip during collection.
collect_ignore = [
    "test_component.py",
    "run_test.py",
    "manual_audio_test.py",
    "vision/test_vision_component.py",
    "components/audio/test_audio_component.py",
]

_HEAVY_LIBS = {
    "torch", "torchvision", "torchaudio", "cv2", "ultralytics", "face_recognition",
    "mediapipe", "pyaudio", "whisper", "TTS", "aiohttp", "sounddevice", "scipy",
    "sklearn", "librosa", "soundfile", "numpy", "pandas", "PIL",
    # Web deps: stubbed only if not installed (CI installs them from requirements-dev.txt)
    "fastapi", "uvicorn",
}


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, roots):
        self._roots = roots

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self._roots and fullname not in sys.modules:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        module = MagicMock(name=spec.name)
        module.__path__ = []  # mark as package so submodule imports resolve
        return module

    def exec_module(self, module):
        pass


_missing = set()
for _name in _HEAVY_LIBS:
    try:
        __import__(_name)
    except Exception:
        _missing.add(_name)

if _missing:
    sys.meta_path.insert(0, _StubFinder(_missing))
