import asyncio

from jarvis.components.conversation import ConversationComponent, LLMError
from jarvis.core.jarvis_core import JarvisCore
from jarvis.core.message import Message


class FakeLLM:
    def __init__(self, reply="Il fait beau aujourd'hui.", models=("qwen2.5:3b",), fail=False):
        self.reply = reply
        self.models = list(models)
        self.fail = fail
        self.calls = []
        self.closed = False

    async def list_models(self):
        if self.fail:
            raise LLMError("offline")
        return self.models

    async def pull(self, model):
        self.models.append(model)

    async def chat(self, model, messages, options=None):
        self.calls.append((model, messages))
        if self.fail:
            raise LLMError("offline")
        return self.reply

    async def close(self):
        self.closed = True


def make(config=None, **llm_kwargs):
    llm = FakeLLM(**llm_kwargs)
    cfg = {"owner_name": "Viny", "language": "fr"}
    cfg.update(config or {})
    return ConversationComponent(cfg, llm_client=llm), llm


async def test_canned_answer_skips_llm_and_personalises_for_owner():
    conv, llm = make()
    await conv.start()
    assert await conv.respond("Jarvis, bonjour") == "Bonjour ! Comment puis-je vous aider ?"
    assert await conv.respond("bonjour Viny") != ""  # name spoken -> not an exact canned match
    await conv.handle_message(Message("vision", "vision_update",
                                      {"faces": [{"identity": "Viny", "confidence": 0.9}]}))
    assert await conv.respond("Jarvis bonjour") == "Bonjour Viny ! Comment puis-je vous aider ?"
    assert all(c[1][-1]["content"] != "bonjour" for c in llm.calls)


async def test_llm_answer_is_cleaned_cached_and_kept_in_history():
    conv, llm = make(reply="**Assistant:** Paris est la capitale. Paris est la capitale. Voilà")
    await conv.start()
    answer = await conv.respond("quelle est la capitale de la France ?")
    assert answer == "Paris est la capitale. Voilà."
    assert conv.state["last_source"] == "llm"
    assert len(conv.history) == 2

    again = await conv.respond("Quelle est la capitale de la France ?")
    assert again == answer and conv.state["last_source"] == "cache"
    assert len(llm.calls) == 1


async def test_vision_query_injects_scene_and_is_not_cached():
    conv, llm = make(reply="Je vois une tasse et Viny.")
    await conv.start()
    await conv.handle_message(Message("vision", "vision_update", {
        "objects": [{"class": "cup"}, {"class": "cup"}, {"class": "laptop"}],
        "faces": [{"identity": "Viny", "confidence": 0.82}],
    }))
    await conv.respond("qu'est-ce que tu vois ?")
    await conv.respond("qu'est-ce que tu vois ?")
    assert len(llm.calls) == 2  # visual answers are never cached
    system = llm.calls[0][1][0]["content"]
    assert "2x cup" in system and "laptop" in system
    assert "Viny 82%" in system and "Viny" in system


async def test_frame_description_uses_vision_model():
    conv, llm = make({"llm": {"vision_model": "llava"}}, reply="Une pièce lumineuse.")
    conv.frame_provider = lambda: "frame"
    conv._encode_frame = lambda frame: "aW1n"
    await conv.start()
    await conv.respond("décris ce que tu vois")
    assert llm.calls[0][0] == "llava"
    assert llm.calls[0][1][0]["images"] == ["aW1n"]
    assert "Une pièce lumineuse." in llm.calls[1][1][0]["content"]


async def test_llm_offline_is_non_fatal():
    conv, _ = make(fail=True)
    await conv.start()  # must not raise
    assert conv.state["llm_available"] is False
    assert await conv.respond("raconte une blague") == "Désolé, je n'arrive pas à réfléchir pour le moment."
    assert await conv.respond("merci") == "À votre service !"


async def test_handle_speech_requires_wake_word_and_speaks():
    conv, llm = make()
    spoken = []

    async def speaker(text):
        spoken.append(text)

    conv.speaker = speaker
    conv.speech_cooldown = 0
    await conv.start()
    assert await conv.handle_speech("quelle heure est-il") is None
    assert await conv.handle_speech("Jarvis, quelle heure est-il") == "Il fait beau aujourd'hui."
    assert spoken == ["Il fait beau aujourd'hui."]
    assert llm.calls[-1][1][-1]["content"] == "quelle heure est-il"


async def test_core_routes_free_speech_to_conversation_but_not_device_commands():
    core = JarvisCore("config/does_not_exist.json")
    conv, llm = make()
    await conv.start()
    routed = []

    async def fake_handle_speech(text):
        routed.append(text)

    conv.handle_speech = fake_handle_speech

    class Automation:
        def _parse_speech_command(self, text):
            return {"device_id": "tv"} if "turn on" in text else None

    core.components.update({"conversation": conv, "automation": Automation()})
    await core._handle_speech_event({"text": "Jarvis, raconte-moi une histoire"})
    await core._handle_speech_event({"text": "jarvis turn on the tv"})
    await asyncio.sleep(0)
    assert routed == ["Jarvis, raconte-moi une histoire"]


def test_clean_response_limits_length():
    long = " ".join(f"Phrase numéro {i}." for i in range(20))
    assert ConversationComponent.clean_response(long).count(".") == 3
    assert ConversationComponent.clean_response(long, long_form=True).count(".") == 8
    assert ConversationComponent.clean_response("", fallback="x") == "x"
