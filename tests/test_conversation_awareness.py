import time
from datetime import datetime, timedelta

from jarvis.components.conversation import ConversationComponent
from jarvis.components.conversation import house_context
from tests.test_conversation import FakeLLM


class FakeSearch:
    available = True

    def __init__(self):
        self.queries = []

    def is_search_query(self, text):
        return "aujourd'hui" in text.lower() or "météo" in text.lower()

    async def search(self, query):
        self.queries.append(query)
        return [{"title": "t", "url": "https://example.org", "snippet": "s"}]

    async def close(self):
        pass


NOW = time.time()

SNAPSHOT = {
    "frigate_online": True,
    "mqtt_connected": True,
    "home_assistant_connected": True,
    "areas": {"salon": {"person": 2}, "allee": {"person": 0, "car": 1}},
    "active_objects": [],
    "people": {"Alice": {"name": "Alice", "camera": "entree", "zones": ["porche"],
                         "last_seen": NOW - 120}},
    "vehicles": {"AB123CD": {"plate": "AB123CD", "name": "Voiture de Bob", "camera": "allee",
                             "last_seen": NOW - 3 * 3600}},
    "home_presence": {"person.bob": {"state": "not_home", "name": "Bob"}},
}


def _history(subject, limit):
    ts = (datetime.now() - timedelta(minutes=2)).isoformat()
    events = [
        {"timestamp": ts, "event_type": "person_identified", "subject": "Alice",
         "camera": "entree", "data": {"name": "Alice", "camera": "entree", "zones": ["porche"]}},
        {"timestamp": ts, "event_type": "home_presence_update", "subject": "person.bob",
         "camera": None, "data": {"name": "Bob", "state": "not_home"}},
    ]
    return [e for e in events if subject in (None, e["subject"])][:limit]


def make(search=None):
    llm = FakeLLM(reply="Alice est passée par l'entrée il y a deux minutes.")
    conv = ConversationComponent({"language": "fr", "web_search": {"enabled": False}},
                                 llm_client=llm, web_search=search)
    calls = []

    async def history(subject, limit):
        calls.append((subject, limit))
        return _history(subject, limit)

    conv.awareness_provider = lambda: SNAPSHOT
    conv.history_provider = history
    return conv, llm, calls


def test_house_query_detection():
    assert house_context.is_house_query("qui est à la maison ?", SNAPSHOT, "fr")
    assert house_context.is_house_query("qui est passé aujourd'hui ?", SNAPSHOT, "fr")
    assert house_context.is_house_query("Alice est rentrée ?", SNAPSHOT, "fr")         # known person
    assert house_context.is_house_query("il y a quelqu'un dans le salon ?", SNAPSHOT, "fr")
    assert house_context.is_house_query("who is home?", SNAPSHOT, "en")
    # News and weather must keep going to the web
    assert not house_context.is_house_query("qu'est-ce qui s'est passé hier en Ukraine", SNAPSHOT, "fr")
    assert not house_context.is_house_query("il fait froid dehors ?", SNAPSHOT, "fr")
    assert not house_context.is_house_query("prix d'une voiture électrique", SNAPSHOT, "fr")


def test_describe_house_fr():
    text = house_context.describe_house(SNAPSHOT, _history(None, 8), "fr", now=NOW)
    assert "Personnes détectées en ce moment : salon (2)." in text
    assert "car — allee (1)" in text
    assert "Alice : vu(e) il y a 2 min, caméra entree, zone porche." in text
    assert "Véhicule AB123CD (Voiture de Bob) : vu il y a 3 h, caméra allee." in text
    assert "Bob (Home Assistant) : pas à la maison." in text
    assert "Alice reconnu(e) sur la caméra entree" in text


def test_describe_house_when_cameras_down():
    text = house_context.describe_house({"mqtt_connected": False}, [], "fr", now=NOW)
    assert text == "Les caméras de la maison ne sont pas disponibles en ce moment."


async def test_house_question_uses_awareness_not_web_and_is_not_cached():
    search = FakeSearch()
    conv, llm, calls = make(search)
    await conv.start()

    await conv.respond("Jarvis, qui est passé aujourd'hui ?")
    assert search.queries == []                       # "aujourd'hui" would normally hit the web
    system = llm.calls[-1][1][0]["content"]
    assert "ÉTAT DE LA MAISON" in system and "Alice" in system
    assert calls == [(None, 8)]                       # no subject named -> recent history
    assert conv.state["last_source"] == "house"

    await conv.respond("Jarvis, qui est passé aujourd'hui ?")
    assert len(llm.calls) == 2                        # never served from cache


async def test_named_subject_fetches_its_history():
    conv, llm, calls = make()
    await conv.start()
    await conv.respond("Jarvis, Alice est rentrée ?")
    assert calls == [("Alice", 5)]


async def test_other_questions_unchanged():
    search = FakeSearch()
    conv, llm, calls = make(search)
    await conv.start()
    await conv.respond("Jarvis, quelle météo demain ?")
    assert search.queries == ["quelle météo demain ?"]
    assert "ÉTAT DE LA MAISON" not in llm.calls[-1][1][0]["content"]
    assert calls == []


async def test_without_awareness_nothing_changes():
    conv = ConversationComponent({"language": "fr", "web_search": {"enabled": False}},
                                 llm_client=FakeLLM())
    await conv.start()
    await conv.respond("Jarvis, qui est à la maison ?")
    assert conv.state["last_source"] == "llm"


def test_core_wires_awareness_into_conversation():
    from jarvis.core.jarvis_core import JarvisCore

    class _Awareness:
        def snapshot(self):
            return SNAPSHOT

    class _Memory:
        async def get_awareness_history(self, subject=None, limit=50):
            return []

    core = JarvisCore("config/does_not_exist.json")
    conv, _, _ = make()
    conv.awareness_provider = conv.history_provider = None
    awareness, memory = _Awareness(), _Memory()
    core.components = {"conversation": conv, "awareness": awareness, "memory": memory}
    core._wire_conversation()
    assert conv.awareness_provider() is SNAPSHOT
    assert conv.history_provider == memory.get_awareness_history
