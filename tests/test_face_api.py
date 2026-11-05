"""Face library REST endpoints (Frigate-style), with a fake detector."""

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import jarvis.web.server as srv  # noqa: E402
from tests.test_face_library import FakeFrame, make_library  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    lib, rec = make_library(tmp_path)

    class Vision:
        face_library = lib

        async def enroll_face(self, name, samples=5):
            lib.validate_name(name)
            for _ in range(samples):
                await lib.register_frame(name, FakeFrame(1))
            return samples

    class Jarvis:
        components = {"vision": Vision()}

    monkeypatch.setattr(srv, "jarvis", Jarvis())
    srv.app.dependency_overrides[srv.security.get_current_user] = lambda: {"username": "t"}
    yield TestClient(srv.app), lib, rec
    srv.app.dependency_overrides.clear()


def test_face_api_flow(client):
    c, lib, rec = client
    assert c.post("/api/faces/Viny/create").json() == {"status": "success", "name": "Viny"}

    r = c.post("/api/faces/Viny/register", files={"file": ("a.jpg", b"1", "image/jpeg")})
    assert r.status_code == 200
    image = r.json()["file"]
    r = c.post("/api/faces/Viny/register", files={"file": ("b.jpg", b"2", "image/jpeg")})
    assert r.status_code == 400 and "Several faces" in r.json()["detail"]

    assert c.post("/api/faces/Viny/enroll", json={"samples": 2}).json()["samples"] == 2
    assert len(c.get("/api/faces").json()["people"]["Viny"]) == 3
    assert c.get(f"/api/faces/Viny/{image}").content == b"jpg"

    lib.record_attempts(FakeFrame(), [{"identity": "unknown", "location": (0, 1, 1, 0)}], now=1.0)
    attempt = c.get("/api/faces").json()["attempts"][0]["file"]
    assert c.get(f"/api/faces/train/{attempt}").status_code == 200
    assert c.post("/api/faces/train/Marie/classify", json={"training_file": attempt}).status_code == 200
    assert "Marie" in c.get("/api/faces").json()["people"]

    assert c.put("/api/faces/Marie/rename", json={"new_name": "Marie Claire"}).json()["name"] == "Marie Claire"
    assert c.post("/api/faces/Viny/delete", json={"ids": [image]}).json()["deleted"] == 1
    assert c.delete("/api/faces/Marie Claire").status_code == 200
    assert c.delete("/api/faces/Marie Claire").status_code == 404


def test_face_api_rejects_bad_input(client):
    c, _, _ = client
    assert c.post("/api/faces/train/create").status_code in (400, 405)  # reserved name
    assert c.get("/api/faces/Viny/..%2F..%2Fsecret.jpg").status_code in (400, 404)
    assert c.post("/api/faces/train/Viny/classify", json={"training_file": "../x.jpg"}).status_code == 400
    big = b"0" * (10 * 1024 * 1024 + 1)
    r = c.post("/api/faces/Viny/register", files={"file": ("x.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_face_api_unavailable_without_face_recognition(monkeypatch):
    class Jarvis:
        components = {"vision": object()}

    monkeypatch.setattr(srv, "jarvis", Jarvis())
    srv.app.dependency_overrides[srv.security.get_current_user] = lambda: {"username": "t"}
    try:
        assert TestClient(srv.app).get("/api/faces").status_code == 503
    finally:
        srv.app.dependency_overrides.clear()
