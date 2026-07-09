import json

from jarvis.settings import get_settings


def test_uvicorn_kwargs_safe_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(tmp_path / "missing.json"))
    for var in ("JARVIS_HOST", "JARVIS_PORT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("JARVIS_RELOAD", "false")

    kw = get_settings().uvicorn_kwargs()
    assert kw["host"] == "127.0.0.1"
    assert kw["port"] == 8000
    assert kw["reload"] is False


def test_uvicorn_kwargs_from_config(monkeypatch, tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"web": {"host": "0.0.0.0", "port": 9000}}))
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg))
    for var in ("JARVIS_HOST", "JARVIS_PORT"):
        monkeypatch.delenv(var, raising=False)

    kw = get_settings().uvicorn_kwargs()
    assert kw["host"] == "0.0.0.0"
    assert kw["port"] == 9000


def test_env_overrides_config(monkeypatch, tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"web": {"host": "0.0.0.0", "port": 9000}}))
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg))
    monkeypatch.setenv("JARVIS_HOST", "127.0.0.1")
    monkeypatch.setenv("JARVIS_PORT", "1234")

    kw = get_settings().uvicorn_kwargs()
    assert kw["host"] == "127.0.0.1"
    assert kw["port"] == 1234
