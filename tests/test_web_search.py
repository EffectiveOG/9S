from jarvis.components.conversation import ConversationComponent, WebSearch
from tests.test_conversation import FakeLLM


class FakeSearch:
    available = True

    def __init__(self, results=None, fail=False):
        self.results = results if results is not None else [{
            "title": "Gabon : la CAN 2027",
            "url": "https://example.org/can",
            "snippet": "Le Gabon se qualifie pour la CAN.",
            "source": "example.org",
            "date": "2026-11-01",
        }]
        self.fail = fail
        self.queries = []
        self.closed = False

    def is_search_query(self, text):
        return WebSearch({}, "fr").is_search_query(text)

    async def search(self, query, max_results=None):
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("network down")
        return self.results

    async def close(self):
        self.closed = True


def make(search, reply="Le Gabon est qualifié pour la CAN, selon example.org."):
    llm = FakeLLM(reply=reply)
    conv = ConversationComponent({"language": "fr"}, llm_client=llm, web_search=search)
    return conv, llm


def test_search_triggers_fr_and_en():
    fr = WebSearch({}, "fr")
    assert fr.is_search_query("quelles sont les dernières nouvelles du Gabon")
    assert fr.is_search_query("quel temps fera-t-il demain ? donne-moi la météo")
    assert fr.is_search_query("qui a gagné la CAN 2025")
    assert not fr.is_search_query("quelle heure est-il")  # Ciel searched the web for this
    assert not fr.is_search_query("raconte-moi une blague")
    en = WebSearch({}, "en")
    assert en.is_search_query("what's the latest news about Apple")
    assert not en.is_search_query("tell me a joke")


def test_google_requires_both_credentials_and_ignores_placeholders(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
    assert not WebSearch({"google_api_key": "${GOOGLE_API_KEY}", "google_cse_id": "x"}).google_enabled
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CSE_ID", "cx")
    ws = WebSearch({})
    assert ws.google_enabled
    assert not WebSearch({"provider": "duckduckgo"}).google_enabled


async def test_google_results_are_parsed_and_fallback_to_duckduckgo(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CSE_ID", "cx")
    ws = WebSearch({"max_results": 2})
    seen = {}

    async def fake_get_json(url, params):
        seen.update(params)
        return {"items": [
            {"title": "A", "link": "https://a.example", "snippet": "sa", "displayLink": "a.example"},
            {"title": "B", "link": "https://b.example", "snippet": "sb"},
            {"title": "C", "link": "https://c.example", "snippet": "sc"},
        ]}

    ws._get_json = fake_get_json
    results = await ws.search("prix du pétrole")
    assert [r["title"] for r in results] == ["A", "B"]
    assert results[0]["source"] == "a.example"
    assert seen["q"] == "prix du pétrole" and seen["num"] == 2 and seen["dateRestrict"] == "m1"

    # Google failing -> DuckDuckGo
    async def broken(url, params):
        raise RuntimeError("quota exceeded")

    class FakeDDGS:
        def text(self, query, **kwargs):
            assert kwargs["max_results"] == 2
            return [{"title": "D", "href": "https://d.example", "body": "sd"}]

    ws._get_json = broken
    ws._ddgs_cls = FakeDDGS
    results = await ws.search("prix du pétrole")
    assert [(r["title"], r["url"], r["snippet"]) for r in results] == [("D", "https://d.example", "sd")]


async def test_fetch_content_extracts_page_text():
    ws = WebSearch({"fetch_content": True, "provider": "duckduckgo", "max_content_length": 80})

    class FakeDDGS:
        def text(self, query, **kwargs):
            return [{"title": "T", "href": "https://t.example", "body": "b"}]

    async def fake_get_text(url):
        return ("<html><body><nav>menu</nav><script>x()</script><article>"
                + "Le contenu principal de la page est ici. " * 5 + "</article></body></html>")

    ws._ddgs_cls = FakeDDGS
    ws._get_text = fake_get_text
    [result] = await ws.search("q")
    assert result["content"].startswith("Le contenu principal")
    assert "menu" not in result["content"] and "x()" not in result["content"]
    assert len(result["content"]) <= 80


def test_summarize_is_plain_text():
    text = WebSearch.summarize([
        {"title": "T1", "snippet": "S1", "content": "Phrase un. Phrase deux. Phrase trois. Quatre.",
         "source": "src", "date": "2026-11-01", "url": "https://x"},
        {"title": "T2", "snippet": "S2"}, {"title": "T3"}, {"title": "T4"},
    ])
    assert "1. T1" in text and "Extract: Phrase un. Phrase deux. Phrase trois." in text
    assert "(src, 2026-11-01)" in text and "T4" not in text and "**" not in text


async def test_conversation_uses_search_results_and_does_not_cache():
    search = FakeSearch()
    conv, llm = make(search)
    await conv.start()
    q = "quelles sont les dernières nouvelles du Gabon ?"
    answer = await conv.respond(q)
    assert answer == "Le Gabon est qualifié pour la CAN, selon example.org."
    assert conv.state["last_source"] == "web"
    assert conv.state["last_sources"] == ["https://example.org/can"]
    system = llm.calls[0][1][0]["content"]
    assert "RÉSULTATS DE RECHERCHE WEB" in system and "Le Gabon se qualifie" in system

    await conv.respond(q)
    assert len(search.queries) == 2 and len(llm.calls) == 2  # fresh every time

    await conv.stop()
    assert search.closed


async def test_conversation_skips_search_for_other_questions_and_survives_failures():
    search = FakeSearch(fail=True)
    conv, llm = make(search, reply="Voici une blague.")
    await conv.start()
    assert await conv.respond("raconte-moi une blague") == "Voici une blague."
    assert search.queries == []

    # search failure -> plain LLM answer, no crash
    assert await conv.respond("quelle est la météo à Libreville") == "Voici une blague."
    assert search.queries == ["quelle est la météo à Libreville"]
    assert conv.state["last_source"] == "llm" and conv.state["last_sources"] == []
    assert "RÉSULTATS DE RECHERCHE WEB" not in llm.calls[-1][1][0]["content"]
