from jarvis.web.security import SecurityManager


def test_authenticate_user_default_admin():
    sm = SecurityManager()
    assert sm.authenticate_user("admin", "wrong") is None
    user = sm.authenticate_user("admin", "admin")
    assert user is not None
    assert user.username == "admin"


def test_token_create_and_verify():
    sm = SecurityManager()
    token = sm.create_access_token({"sub": "admin"})
    assert sm.verify_token(token) is True
    assert sm.verify_token(None) is False
    assert sm.verify_token("not-a-jwt") is False


def test_password_hash_roundtrip():
    sm = SecurityManager()
    hashed = sm.get_password_hash("s3cret")
    assert sm.verify_password("s3cret", hashed)
    assert not sm.verify_password("nope", hashed)
