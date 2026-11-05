import asyncio
from unittest.mock import MagicMock

import pytest

from jarvis.components.conversation import ConversationComponent
from jarvis.components.vision.face_library import FaceLibrary, FaceLibraryError
from tests.test_conversation import FakeLLM


class FakeRecognizer:
    def __init__(self):
        self.known_faces = {}
        self.reloaded = []

    def add_encoding(self, name, encoding):
        self.known_faces.setdefault(name, []).append(encoding)

    def reload_person(self, name, person_dir):
        self.reloaded.append(name)
        files = list(person_dir.iterdir()) if person_dir.is_dir() else []
        if files:
            self.known_faces[name] = [f.name for f in files]
        else:
            self.known_faces.pop(name, None)

    def remove_person(self, name):
        self.known_faces.pop(name, None)

    def rename_person(self, old, new):
        self.known_faces[new] = self.known_faces.pop(old)


class FakeFrame:
    """Stands in for a numpy image; `faces` is what the detector will find."""

    def __init__(self, faces=1):
        self.faces = faces

    def copy(self):
        return self


def make_library(tmp_path, **kwargs):
    rec = FakeRecognizer()
    lib = FaceLibrary(rec, faces_dir=str(tmp_path / "known"),
                      attempts_dir=str(tmp_path / "attempts"), **kwargs)
    lib._detect = lambda frame: [((10, 60, 60, 10), f"enc{i}") for i in range(frame.faces)]
    lib._crop = lambda frame, loc: frame
    lib._write_jpeg = lambda path, image: path.write_bytes(b"jpg")
    lib._decode = lambda data: FakeFrame(int(data.decode() or 0)) if data != b"jpg" else FakeFrame(1)
    return lib, rec


def test_names_and_files_are_validated(tmp_path):
    lib, _ = make_library(tmp_path)
    for bad in ["", "../etc", "a/b", "train", "unknown", ".hidden", "x" * 70, "_x"]:
        with pytest.raises(FaceLibraryError):
            lib.validate_name(bad)
    assert lib.validate_name("  Jean-Paul  Émane ") == "Jean-Paul Émane"
    lib.create_person("Viny")
    for bad in ["../../secret.jpg", "x.txt", ".x.jpg", "a/b.jpg"]:
        with pytest.raises(FaceLibraryError):
            lib.image_path("Viny", bad)


async def test_register_requires_exactly_one_face(tmp_path):
    lib, rec = make_library(tmp_path)
    saved = await lib.register_frame("Viny", FakeFrame(1))
    assert lib.people() == {"Viny": [saved]}
    assert rec.known_faces == {"Viny": ["enc0"]}
    with pytest.raises(FaceLibraryError, match="No face"):
        await lib.register_frame("Viny", FakeFrame(0))
    with pytest.raises(FaceLibraryError, match="Several faces"):
        await lib.register_image("Viny", b"2")
    with pytest.raises(FaceLibraryError):
        await lib.register_frame("../x", FakeFrame(1))
    assert len(lib.people()["Viny"]) == 1


async def test_delete_rename_and_remove_people(tmp_path):
    lib, rec = make_library(tmp_path)
    first = await lib.register_frame("Viny", FakeFrame(1))
    await asyncio.sleep(0.002)
    await lib.register_frame("Viny", FakeFrame(1))
    assert await lib.delete_images("Viny", [first]) == 1
    assert rec.reloaded == ["Viny"] and len(lib.people()["Viny"]) == 1

    assert await lib.rename_person("Viny", "Viny Ndolo") == "Viny Ndolo"
    assert list(lib.people()) == ["Viny Ndolo"] and "Viny Ndolo" in rec.known_faces
    lib.create_person("Marie")
    with pytest.raises(FaceLibraryError, match="already exists"):
        await lib.rename_person("Marie", "Viny Ndolo")

    assert await lib.delete_person("Viny Ndolo") is True
    assert await lib.delete_person("Viny Ndolo") is False
    assert "Viny Ndolo" not in rec.known_faces and list(lib.people()) == ["Marie"]


async def test_attempts_are_throttled_capped_and_trainable(tmp_path):
    lib, rec = make_library(tmp_path, max_attempts=3, attempt_interval=5)
    unknown = {"identity": "unknown", "confidence": 0.0, "location": (0, 1, 1, 0)}
    viny = {"identity": "Viny", "confidence": 0.82, "location": (0, 1, 1, 0)}
    lib.record_attempts(FakeFrame(), [unknown, viny], now=1000.0)
    lib.record_attempts(FakeFrame(), [unknown], now=1002.0)  # throttled
    assert sorted((a["identity"], a["score"]) for a in lib.attempts()) == [("Viny", 0.82), ("unknown", 0.0)]

    for t in (1010.0, 1020.0, 1030.0):
        lib.record_attempts(FakeFrame(), [unknown], now=t)
    attempts = lib.attempts()
    assert len(attempts) == 3 and attempts[0]["timestamp"] == 1030.0  # oldest pruned

    saved = await lib.train_attempt(attempts[0]["file"], "Marie")
    assert lib.people() == {"Marie": [saved]} and rec.known_faces["Marie"] == ["enc0"]
    assert len(lib.attempts()) == 2
    assert lib.delete_attempts([a["file"] for a in lib.attempts()]) == 2
    with pytest.raises(FaceLibraryError):
        await lib.train_attempt("../../x.jpg", "Marie")


def test_attempts_disabled(tmp_path):
    lib, _ = make_library(tmp_path, max_attempts=0)
    lib.record_attempts(FakeFrame(), [{"identity": "x", "location": (0, 1, 1, 0)}])
    assert lib.attempts() == []


def test_real_image_helpers(tmp_path):
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    if isinstance(np, MagicMock) or isinstance(cv2, MagicMock):
        pytest.skip("numpy/opencv are stubbed in this environment")
    lib = FaceLibrary(FakeRecognizer(), faces_dir=str(tmp_path / "k"),
                      attempts_dir=str(tmp_path / "a"))
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    crop = lib._crop(frame, (50, 150, 150, 50))  # 100x100 face + 50% margin
    assert crop.shape[:2] == (200, 200)
    path = tmp_path / "x.jpg"
    lib._write_jpeg(path, crop)
    assert lib._decode(path.read_bytes()).shape == (200, 200, 3)
    with pytest.raises(FaceLibraryError):
        lib._decode(b"not an image")


# ------------------------------------------------------------ voice command
def make_conv(owner="Viny", lang="fr"):
    conv = ConversationComponent({"owner_name": owner, "language": lang,
                                  "web_search": {"enabled": False}}, llm_client=FakeLLM())
    calls = []

    async def enroller(name):
        calls.append(name)
        return 0 if name == "Fantôme" else 5

    conv.face_enroller = enroller
    return conv, calls


@pytest.mark.parametrize("text,expected", [
    ("Jarvis, apprends mon visage", "Viny"),
    ("enregistre mon visage s'il te plaît", "Viny"),
    ("Jarvis mémorise le visage de marie", "Marie"),
    ("apprends le visage d'Anne-Sophie", "Anne-Sophie"),
])
async def test_voice_enrollment_fr(text, expected):
    conv, calls = make_conv()
    await conv.start()
    answer = await conv.respond(text)
    assert calls == [expected]
    assert conv.state["last_source"] == "face_enroll"
    assert "5 images" in answer and expected in answer


async def test_voice_enrollment_en_and_edge_cases():
    conv, calls = make_conv(lang="en")
    await conv.start()
    await conv.respond("Jarvis, learn my face")
    await conv.respond("remember Tom's face")
    await conv.respond("learn the face of Sarah")
    assert calls == ["Viny", "Tom", "Sarah"]

    conv, calls = make_conv(owner=None)
    await conv.start()
    assert "prénom" in await conv.respond("apprends mon visage")  # nobody to name
    assert calls == []
    assert "Fantôme" in await conv.respond("apprends le visage de Fantôme")  # 0 samples
    assert "pas réussi" in conv.state["last_response"]

    conv.face_enroller = None
    assert await conv.respond("apprends le visage de Paul") == "La reconnaissance faciale n'est pas activée."
    # an ordinary question mentioning a face is not an enrollment
    conv2, calls2 = make_conv()
    await conv2.start()
    await conv2.respond("tu as vu mon visage ?")
    assert calls2 == []


async def test_voice_enrollment_announces_before_capturing():
    conv, _ = make_conv()
    spoken = []

    async def speaker(text):
        spoken.append(text)

    conv.speaker = speaker
    conv.speech_cooldown = 0
    await conv.start()
    await conv.handle_speech("Jarvis, apprends mon visage")
    assert spoken[0] == "Regardez la caméra, je mémorise le visage de Viny."
    assert spoken[1].startswith("C'est fait : 5 images")


# ------------------------------------------------------- camera enrollment
async def test_vision_enroll_face_skips_bad_frames(tmp_path):
    from jarvis.components.vision.vision_component import VisionComponent

    vision = VisionComponent.__new__(VisionComponent)  # skip camera/model setup
    lib, rec = make_library(tmp_path)
    vision.face_library = lib
    vision.enroll_samples, vision.enroll_timeout = 3, 2.0
    frames = iter([FakeFrame(0), FakeFrame(2), FakeFrame(1), FakeFrame(1), FakeFrame(1)])

    class Feed:
        def __get__(self, obj, owner=None):
            return next(frames, None)

    vision.__class__ = type("V", (VisionComponent,), {"latest_frame": Feed()})
    saved = await vision.enroll_face("Viny", interval=0)
    assert saved == 3 and len(rec.known_faces["Viny"]) == 3
    with pytest.raises(FaceLibraryError):
        await vision.enroll_face("../x")


async def test_same_millisecond_registrations_do_not_overwrite(tmp_path, monkeypatch):
    import jarvis.components.vision.face_library as fl
    monkeypatch.setattr(fl.time, "time", lambda: 1000.0)
    lib, _ = make_library(tmp_path)
    names = {await lib.register_frame("Viny", FakeFrame(1)) for _ in range(5)}
    assert len(names) == 5 and len(lib.people()["Viny"]) == 5
