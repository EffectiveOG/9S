# jarvis/components/vision/face_library.py
"""Face library: enrollment, training and management of known faces.

Combines two ideas:
  * Ciel's voice enrollment ("apprends mon visage"): capture several samples
    of the person in front of the camera and add them to the known faces;
  * Frigate's face library: one folder per person under `known_faces_dir`
    (already the layout FaceRecognizer loads), images can be uploaded or
    deleted, people renamed or removed, and recent recognition attempts are
    kept in a "train" folder so an unknown (or wrongly recognised) face can
    later be assigned to a person with one click.

All file operations are validated (no path traversal) and every encoding is
computed off the event loop.
"""

import asyncio
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...utils.logging_utils import get_logger

logger = get_logger(__name__)

_IMAGE_EXT = (".jpg", ".jpeg", ".png")
_NAME_RE = re.compile(r"^[^\W_][\w .'-]{0,63}$", re.UNICODE)
_FILE_RE = re.compile(r"^[\w.-]{1,128}$")
_ATTEMPT_RE = re.compile(r"^(?P<ts>\d+)-(?P<identity>.+)-(?P<score>\d\.\d\d)\.jpg$")
RESERVED_NAMES = {"train", "attempts", "unknown"}


class FaceLibraryError(ValueError):
    """Invalid request (bad name, no face, several faces, missing file...)."""


class FaceLibrary:
    def __init__(self, recognizer, faces_dir: str = "data/known_faces",
                 attempts_dir: str = "data/face_attempts", max_attempts: int = 50,
                 attempt_interval: float = 5.0, crop_margin: float = 0.5):
        self.recognizer = recognizer
        self.faces_dir = Path(faces_dir)
        self.attempts_dir = Path(attempts_dir)
        self.max_attempts = max(0, int(max_attempts))
        self.attempt_interval = float(attempt_interval)
        self.crop_margin = float(crop_margin)
        self._last_attempt: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        if self.max_attempts:
            self.attempts_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------- validation
    @staticmethod
    def validate_name(name: str) -> str:
        name = " ".join(str(name or "").split())
        if not _NAME_RE.match(name) or ".." in name or name.lower() in RESERVED_NAMES:
            raise FaceLibraryError(f"Invalid person name: {name!r}")
        return name

    @staticmethod
    def _validate_file(filename: str) -> str:
        if not _FILE_RE.match(filename or "") or filename.startswith(".") \
                or not filename.lower().endswith(_IMAGE_EXT):
            raise FaceLibraryError(f"Invalid file name: {filename!r}")
        return filename

    def _person_dir(self, name: str) -> Path:
        return self.faces_dir / self.validate_name(name)

    def image_path(self, name: str, filename: str) -> Path:
        path = self._person_dir(name) / self._validate_file(filename)
        if not path.is_file():
            raise FaceLibraryError("Image not found")
        return path

    def attempt_path(self, filename: str) -> Path:
        path = self.attempts_dir / self._validate_file(filename)
        if not path.is_file():
            raise FaceLibraryError("Attempt not found")
        return path

    # -------------------------------------------- image helpers (overridable)
    def _detect(self, frame) -> List[Tuple[Tuple[int, int, int, int], Any]]:
        return self.recognizer.detect(frame)

    @staticmethod
    def _decode(data: bytes):
        import cv2
        import numpy as np
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise FaceLibraryError("Unsupported or corrupt image")
        return frame

    @staticmethod
    def _write_jpeg(path: Path, image):
        import cv2
        if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            raise OSError(f"Could not write {path}")

    def _crop(self, frame, location):
        """Crop a face with a margin, so the saved image can be re-encoded later."""
        top, right, bottom, left = location
        h, w = frame.shape[:2]
        mh, mw = int((bottom - top) * self.crop_margin), int((right - left) * self.crop_margin)
        return frame[max(0, top - mh):min(h, bottom + mh), max(0, left - mw):min(w, right + mw)]

    @staticmethod
    def _new_filename() -> str:
        # timestamp for ordering + random suffix: two images registered in the
        # same millisecond (fast enrollment, parallel uploads) must not collide
        return f"{int(time.time() * 1000)}-{secrets.token_hex(3)}.jpg"

    # -------------------------------------------------------------- queries
    def people(self) -> Dict[str, List[str]]:
        result = {}
        for d in sorted(self.faces_dir.iterdir()) if self.faces_dir.is_dir() else []:
            if d.is_dir() and not d.name.startswith("."):
                result[d.name] = sorted(f.name for f in d.iterdir()
                                        if f.suffix.lower() in _IMAGE_EXT)
        return result

    def attempts(self) -> List[Dict[str, Any]]:
        items = []
        if self.attempts_dir.is_dir():
            for f in self.attempts_dir.iterdir():
                m = _ATTEMPT_RE.match(f.name)
                if m:
                    items.append({"file": f.name, "identity": m["identity"],
                                  "score": float(m["score"]),
                                  "timestamp": int(m["ts"]) / 1000})
        return sorted(items, key=lambda a: a["timestamp"], reverse=True)

    # ------------------------------------------------------------- mutations
    def create_person(self, name: str) -> str:
        path = self._person_dir(name)
        path.mkdir(parents=True, exist_ok=True)
        return path.name

    async def register_frame(self, name: str, frame) -> str:
        """Add the single face found in `frame` (BGR) to `name`'s images."""
        person = self._person_dir(name)  # validate before the expensive part
        detections = await asyncio.to_thread(self._detect, frame)
        if not detections:
            raise FaceLibraryError("No face found in the image")
        if len(detections) > 1:
            raise FaceLibraryError("Several faces found; use an image with one face")
        location, encoding = detections[0]
        async with self._lock:
            person.mkdir(parents=True, exist_ok=True)
            filename = self._new_filename()
            await asyncio.to_thread(self._write_jpeg, person / filename,
                                    self._crop(frame, location))
            self.recognizer.add_encoding(person.name, encoding)
        logger.info(f"Registered face image {filename} for {person.name}")
        return filename

    async def register_image(self, name: str, data: bytes) -> str:
        frame = await asyncio.to_thread(self._decode, data)
        return await self.register_frame(name, frame)

    async def train_attempt(self, filename: str, name: str) -> str:
        """Frigate's "train": assign a recent attempt to a person."""
        src = self.attempt_path(filename)
        frame = await asyncio.to_thread(self._decode, src.read_bytes())
        saved = await self.register_frame(name, frame)
        src.unlink(missing_ok=True)
        return saved

    async def delete_images(self, name: str, filenames: List[str]) -> int:
        person = self._person_dir(name)
        removed = 0
        async with self._lock:
            for filename in filenames:
                path = person / self._validate_file(filename)
                if path.is_file():
                    path.unlink()
                    removed += 1
            if removed:
                await asyncio.to_thread(self.recognizer.reload_person, person.name, person)
        return removed

    async def delete_person(self, name: str) -> bool:
        person = self._person_dir(name)
        async with self._lock:
            if not person.is_dir():
                return False
            await asyncio.to_thread(shutil.rmtree, person)
            self.recognizer.remove_person(person.name)
        return True

    async def rename_person(self, old: str, new: str) -> str:
        src, dst = self._person_dir(old), self._person_dir(new)
        async with self._lock:
            if not src.is_dir():
                raise FaceLibraryError(f"Unknown person: {old}")
            if dst.exists():
                raise FaceLibraryError(f"{new} already exists")
            src.rename(dst)
            self.recognizer.rename_person(src.name, dst.name)
        return dst.name

    def delete_attempts(self, filenames: List[str]) -> int:
        removed = 0
        for filename in filenames:
            path = self.attempts_dir / self._validate_file(filename)
            if path.is_file():
                path.unlink()
                removed += 1
        return removed

    # ------------------------------------------------------------- attempts
    def record_attempts(self, frame, faces: List[Dict[str, Any]], now: Optional[float] = None):
        """Blocking: keep a crop of recent recognitions for later training.

        Throttled per identity (one crop every `attempt_interval` seconds) and
        capped at `max_attempts` files, oldest removed first.
        """
        if not self.max_attempts or frame is None:
            return
        now = time.time() if now is None else now
        for face in faces or []:
            identity = str(face.get("identity") or "unknown")
            if not _NAME_RE.match(identity):
                identity = "unknown"
            if now - self._last_attempt.get(identity, float("-inf")) < self.attempt_interval:
                continue
            self._last_attempt[identity] = now
            score = max(0.0, min(0.99, float(face.get("confidence") or 0.0)))
            name = f"{int(now * 1000)}-{identity}-{score:.2f}.jpg"
            try:
                self._write_jpeg(self.attempts_dir / name, self._crop(frame, face["location"]))
            except Exception as e:
                logger.debug(f"Could not save face attempt: {e}")
        self._prune_attempts()

    def _prune_attempts(self):
        files = sorted((f for f in self.attempts_dir.iterdir() if _ATTEMPT_RE.match(f.name)),
                       key=lambda f: f.name)
        for f in files[:max(0, len(files) - self.max_attempts)]:
            f.unlink(missing_ok=True)
