# jarvis/components/conversation/conversation_component.py
"""Conversational brain for Jarvis, ported from the Ciel project.

What it brings over from Ciel's ConversationModule:
  * a local LLM (Ollama, e.g. qwen2.5:3b) for free-form questions;
  * instant canned answers for greetings / identity questions;
  * a response cache for repeated non-visual questions;
  * vision-aware answers: the latest `vision_update` (objects, recognised
    faces, gestures) is injected into the prompt, and an optional multimodal
    model (e.g. llava) can describe the current camera frame - Ciel's LLaVA
    "deep analysis";
  * owner recognition: when the configured owner's face is recognised (or
    their name is spoken) the assistant addresses them personally - Ciel's
    "Master Viny" feature, made configurable instead of hard-coded;
  * TTS-friendly response cleaning and echo prevention while speaking.

Unlike Ciel (threads + blocking calls) everything here is asyncio-native and
fits 9S's BaseComponent / Message bus model.
"""

import asyncio
import base64
import re
import time
from collections import Counter, OrderedDict, deque
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional

from ...core.base_component import BaseComponent
from ...core.message import Message
from ...utils.logging_utils import get_logger
from .llm_client import LLMError, OllamaClient

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Language resources
# ---------------------------------------------------------------------------

_TEXTS = {
    "fr": {
        "system": (
            "Tu es {name}, un assistant vocal domestique intelligent qui voit "
            "grâce à une caméra et contrôle la maison. Réponds dans la langue de "
            "l'utilisateur, en phrases courtes et naturelles, sans markdown ni "
            "listes, car ta réponse sera lue à voix haute. Pour une question "
            "visuelle, parle à la première personne (« Je vois… ») et n'invente "
            "rien qui ne figure pas dans la description de la scène."
        ),
        "owner": "Tu parles à {owner}, ton créateur. Adresse-toi à lui par son prénom.",
        "scene": "DESCRIPTION DE LA SCÈNE ACTUELLE :\n{scene}",
        "no_camera": "La caméra n'est pas disponible pour le moment.",
        "error": "Désolé, je n'arrive pas à réfléchir pour le moment.",
        "empty": "Je vous écoute.",
        "describe": "Décris précisément cette image en deux ou trois phrases, en français.",
        "vision_keywords": [
            "vois", "voir", "regarde", "montre", "décris", "décrire",
            "qu'est-ce que je tiens", "devant toi", "caméra",
        ],
        "canned": {
            "comment tu t'appelles": "Je m'appelle {name}.",
            "quel est ton nom": "Je m'appelle {name}.",
            "qui es-tu": "Je suis {name}, votre assistant.",
            "bonjour": "Bonjour{owner_suffix} ! Comment puis-je vous aider ?",
            "bonsoir": "Bonsoir{owner_suffix} !",
            "salut": "Salut{owner_suffix} !",
            "comment ça va": "Tout va bien, merci{owner_suffix}.",
            "merci": "À votre service{owner_suffix} !",
            "au revoir": "Au revoir{owner_suffix}, à bientôt !",
            "bonne nuit": "Bonne nuit{owner_suffix} !",
        },
    },
    "en": {
        "system": (
            "You are {name}, a smart home voice assistant that can see through a "
            "camera and control the house. Answer in the user's language, in "
            "short natural sentences with no markdown or lists, because your "
            "answer is read aloud. For visual questions, speak in the first "
            "person (\"I see...\") and never invent anything that is not in the "
            "scene description."
        ),
        "owner": "You are talking to {owner}, your creator. Address them by name.",
        "scene": "CURRENT SCENE DESCRIPTION:\n{scene}",
        "no_camera": "The camera is not available right now.",
        "error": "Sorry, I can't think right now.",
        "empty": "I'm listening.",
        "describe": "Describe this image precisely in two or three sentences.",
        "vision_keywords": [
            "see", "look", "show", "describe", "what am i holding",
            "in front of you", "camera",
        ],
        "canned": {
            "what's your name": "My name is {name}.",
            "what is your name": "My name is {name}.",
            "who are you": "I'm {name}, your assistant.",
            "hello": "Hello{owner_suffix}! How can I help?",
            "good evening": "Good evening{owner_suffix}!",
            "how are you": "All systems are fine, thank you{owner_suffix}.",
            "thank you": "You're welcome{owner_suffix}!",
            "goodbye": "Goodbye{owner_suffix}!",
            "good night": "Good night{owner_suffix}!",
        },
    },
}

_ROLE_PREFIX = re.compile(
    r"^\s*(utilisateur|user|humain|human|assistant|ia|ai|jarvis|ciel)\s*:\s*",
    re.IGNORECASE,
)
_MARKDOWN = re.compile(r"(\*\*|__|\*|`+|^#+\s*|^\s*[-•]\s+)", re.MULTILINE)
_SENTENCE = re.compile(r"[^.!?]+[.!?]*")


class ConversationComponent(BaseComponent):
    """LLM-backed conversation component (local Ollama)."""

    def __init__(self, config: Dict[str, Any], llm_client: Optional[Any] = None):
        super().__init__("conversation")
        self.config = config or {}
        llm_cfg = self.config.get("llm", {}) or {}

        self.assistant_name: str = self.config.get("assistant_name", "Jarvis")
        self.owner_name: Optional[str] = self.config.get("owner_name") or None
        lang = str(self.config.get("language", "fr")).lower()[:2]
        self.language = lang if lang in _TEXTS else "en"
        self.texts = _TEXTS[self.language]

        self.wake_word: str = str(self.config.get("wake_word", "jarvis")).lower()
        self.require_wake_word: bool = bool(self.config.get("require_wake_word", True))

        self.model: str = llm_cfg.get("model", "qwen2.5:3b")
        self.vision_model: Optional[str] = llm_cfg.get("vision_model") or None
        self.auto_pull: bool = bool(llm_cfg.get("auto_pull", False))
        self.options = {
            "temperature": float(llm_cfg.get("temperature", 0.7)),
            "num_predict": int(llm_cfg.get("max_tokens", 200)),
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        }
        self.llm = llm_client or OllamaClient(
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            timeout=float(llm_cfg.get("timeout", 60)),
        )
        self.llm_available = False

        self.history: Deque[Dict[str, str]] = deque(
            maxlen=max(0, int(self.config.get("history_size", 6))) * 2
        )
        self.cache_size = int(self.config.get("cache_size", 200))
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self.speech_cooldown = float(self.config.get("speech_cooldown", 1.0))

        # Hooks wired by JarvisCore (kept optional so the component is testable
        # and usable without audio/vision hardware).
        self.speaker: Optional[Callable[[str], Awaitable[Any]]] = None
        self.frame_provider: Optional[Callable[[], Any]] = None

        self.latest_vision: Dict[str, Any] = {}
        self._latest_vision_ts: float = 0.0
        self._lock = asyncio.Lock()
        self._speaking_until: float = 0.0

    # ------------------------------------------------------------------ life
    async def start(self):
        await super().start()
        try:
            models = await self.llm.list_models()
            if not self._model_present(self.model, models):
                if self.auto_pull:
                    await self.llm.pull(self.model)
                else:
                    logger.warning(
                        f"Ollama model '{self.model}' not found (available: {models}). "
                        f"Run `ollama pull {self.model}` or set llm.auto_pull."
                    )
            self.llm_available = True
        except LLMError as e:
            # Non-fatal: canned answers still work, and we retry on each request.
            logger.warning(f"Conversation LLM unavailable at startup: {e}")
            self.llm_available = False
        self.update_state({"llm_available": self.llm_available, "model": self.model})

    async def stop(self):
        try:
            await self.llm.close()
        except Exception as e:
            logger.debug(f"Error closing LLM client: {e}")
        await super().stop()

    @staticmethod
    def _model_present(model: str, models: List[str]) -> bool:
        # Ollama reports "qwen2.5:3b" or "llava:latest"; accept an untagged name.
        return any(m == model or m.split(":")[0] == model for m in models)

    # ------------------------------------------------------------- bus input
    async def handle_message(self, message: Message):
        """Consume bus messages (vision updates keep the scene context fresh)."""
        if message.message_type == "vision_update":
            self.latest_vision = message.data or {}
            self._latest_vision_ts = time.time()

    @property
    def is_busy(self) -> bool:
        return self._lock.locked() or time.time() < self._speaking_until

    async def handle_speech(self, text: str) -> Optional[str]:
        """Entry point for recognised speech routed by JarvisCore.

        Ignores the utterance when it lacks the wake word (if required), or
        while we're answering / speaking (avoids answering our own voice).
        """
        if not self.is_running or not text:
            return None
        if self.require_wake_word and self.wake_word not in text.lower():
            return None
        if self.is_busy:
            logger.info("Conversation busy; ignoring utterance")
            return None
        return await self.respond(text, speak=True)

    # -------------------------------------------------------------- pipeline
    async def respond(self, text: str, speak: bool = False) -> str:
        """Produce (and optionally speak) an answer to `text`."""
        async with self._lock:
            started = time.time()
            query = self._strip_wake_word(text)
            if not query:
                answer, source = self.texts["empty"], "canned"
            else:
                answer, source = await self._answer(query)

            self.update_state({
                "last_query": query,
                "last_response": answer,
                "last_source": source,
                "last_latency": round(time.time() - started, 3),
                "llm_available": self.llm_available,
            })
            await self.publish(Message(
                sender=self.name,
                message_type="conversation_response",
                data={"query": query, "response": answer, "source": source},
            ))
            if speak:
                await self._speak(answer)
            return answer

    async def _answer(self, query: str):
        owner_present = self._owner_present(query)
        canned = self._canned_answer(query, owner_present)
        if canned:
            return canned, "canned"

        is_visual = self._is_vision_query(query)
        key = self._cache_key(query, owner_present)
        if not is_visual and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key], "cache"

        scene = await self._scene_context() if is_visual else ""
        messages = self._build_messages(query, scene, owner_present)
        try:
            raw = await self.llm.chat(self.model, messages, self.options)
            self.llm_available = True
        except LLMError as e:
            logger.error(f"LLM generation failed: {e}")
            self.llm_available = False
            return self.texts["error"], "error"

        answer = self.clean_response(raw, long_form=is_visual, fallback=self.texts["empty"])
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})
        if not is_visual:
            self._cache_put(key, answer)
        return answer, "llm"

    async def _speak(self, text: str):
        if self.speaker is None:
            return
        # Rough speaking-time estimate (~15 chars/s) + cooldown, so the audio
        # component doesn't feed our own voice back to us (Ciel's echo guard).
        self._speaking_until = time.time() + len(text) / 15.0 + self.speech_cooldown
        try:
            await self.speaker(text)
        except Exception as e:
            logger.error(f"Error speaking response: {e}")
        finally:
            self._speaking_until = time.time() + self.speech_cooldown

    # --------------------------------------------------------------- helpers
    def _strip_wake_word(self, text: str) -> str:
        cleaned = re.sub(rf"\b{re.escape(self.wake_word)}\b[\s,!.?:]*", " ",
                         text, flags=re.IGNORECASE)
        return " ".join(cleaned.split()).strip(" ,")

    def _owner_present(self, query: str) -> bool:
        if not self.owner_name:
            return False
        owner = self.owner_name.lower()
        if owner in query.lower():
            return True
        for face in self.latest_vision.get("faces", []) or []:
            if str(face.get("identity", "")).lower() == owner:
                return True
        return False

    def _canned_answer(self, query: str, owner_present: bool) -> Optional[str]:
        normalized = query.lower().strip(" ?!.")
        suffix = f" {self.owner_name}" if owner_present and self.owner_name else ""
        for pattern, template in self.texts["canned"].items():
            # whole-utterance match only, so "bonjour, quelle heure est-il" goes to the LLM
            if normalized == pattern or normalized.rstrip(" ,") == pattern:
                return template.format(name=self.assistant_name, owner_suffix=suffix)
        return None

    def _is_vision_query(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in self.texts["vision_keywords"])

    @staticmethod
    def _cache_key(query: str, owner_present: bool) -> str:
        return f"{int(owner_present)}|{' '.join(query.lower().split())}"

    def _cache_put(self, key: str, value: str):
        if self.cache_size <= 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def clear_history(self):
        self.history.clear()
        self._cache.clear()

    def _build_messages(self, query: str, scene: str, owner_present: bool) -> List[Dict[str, Any]]:
        system = self.texts["system"].format(name=self.assistant_name)
        if owner_present and self.owner_name:
            system += "\n" + self.texts["owner"].format(owner=self.owner_name)
        extra = self.config.get("system_prompt")
        if extra:
            system += "\n" + str(extra)
        if scene:
            system += "\n\n" + self.texts["scene"].format(scene=scene)
        return [{"role": "system", "content": system}, *self.history,
                {"role": "user", "content": query}]

    # --------------------------------------------------------- vision context
    async def _scene_context(self) -> str:
        parts: List[str] = []
        detected = self.describe_detections(self.latest_vision, self.owner_name)
        if detected:
            parts.append(detected)
        description = await self._describe_frame()
        if description:
            parts.append(description)
        return "\n".join(parts) if parts else self.texts["no_camera"]

    @staticmethod
    def describe_detections(vision: Dict[str, Any], owner_name: Optional[str] = None) -> str:
        """Turn a 9S `vision_update` payload into a compact text description."""
        if not vision:
            return ""
        lines = []
        objects = [o.get("class") for o in vision.get("objects", []) or [] if o.get("class")]
        if objects:
            counts = Counter(objects)
            lines.append("Objects: " + ", ".join(
                f"{n}x {c}" if n > 1 else c for c, n in counts.most_common()))
        faces = vision.get("faces", []) or []
        if faces:
            known = [f for f in faces if f.get("identity") not in (None, "", "unknown")]
            desc = f"Faces: {len(faces)}"
            if known:
                desc += " (recognised: " + ", ".join(
                    f"{f['identity']} {float(f.get('confidence', 0)):.0%}" for f in known) + ")"
            lines.append(desc)
            if owner_name and any(str(f.get("identity", "")).lower() == owner_name.lower()
                                  for f in known):
                lines.append(f"The owner, {owner_name}, is in front of the camera.")
        gestures = []
        for g in vision.get("gestures", []) or []:
            gestures.append(g.get("gesture") or g.get("name") or g.get("type")
                            if isinstance(g, dict) else str(g))
        gestures = [g for g in gestures if g]
        if gestures:
            lines.append("Gestures: " + ", ".join(map(str, gestures)))
        return "\n".join(lines)

    async def _describe_frame(self) -> str:
        """Ask a multimodal model (e.g. llava) to describe the current frame."""
        if not self.vision_model or self.frame_provider is None:
            return ""
        try:
            frame = self.frame_provider()
            if frame is None:
                return ""
            image_b64 = await asyncio.to_thread(self._encode_frame, frame)
            if not image_b64:
                return ""
            return await self.llm.chat(
                self.vision_model,
                [{"role": "user", "content": self.texts["describe"], "images": [image_b64]}],
                {"num_predict": 150, "temperature": 0.2},
            )
        except Exception as e:
            logger.warning(f"Frame description failed: {e}")
            return ""

    @staticmethod
    def _encode_frame(frame) -> Optional[str]:
        import cv2  # heavy dependency, only needed when a vision model is configured
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None

    # ------------------------------------------------------- response cleaning
    @staticmethod
    def clean_response(response: str, long_form: bool = False,
                       fallback: str = "") -> str:
        """Make an LLM answer TTS-friendly (ported from Ciel's _clean_response).

        Strips markdown and role prefixes ("Assistant:"), drops duplicated
        sentences and caps the length (longer for scene descriptions).
        """
        if not response or not response.strip():
            return fallback
        max_sentences, max_chars = (8, 700) if long_form else (3, 300)

        sentences: List[str] = []
        seen = set()
        for line in response.splitlines():
            line = _ROLE_PREFIX.sub("", _MARKDOWN.sub("", line)).strip()
            for sentence in _SENTENCE.findall(line):
                sentence = " ".join(sentence.split())
                norm = sentence.lower().strip(".,!? ")
                if len(norm) < 2 or norm in seen:
                    continue
                seen.add(norm)
                sentences.append(sentence)

        result = ""
        for sentence in sentences[:max_sentences]:
            candidate = f"{result} {sentence}".strip()
            if result and len(candidate) > max_chars:
                break
            result = candidate
        if result and result[-1] not in ".!?":
            result += "."
        return result or fallback
