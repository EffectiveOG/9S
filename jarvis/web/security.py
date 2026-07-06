# jarvis/web/security.py

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
from types import SimpleNamespace
from pathlib import Path
import os
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_or_create_secret() -> str:
    """Use JARVIS_SECRET_KEY if set, else persist a random key to disk so
    issued tokens survive restarts (a fresh per-process key would invalidate
    every token on reload)."""
    env_key = os.getenv("JARVIS_SECRET_KEY")
    if env_key:
        return env_key
    key_path = Path(__file__).resolve().parents[2] / "config" / ".secret_key"
    try:
        if key_path.exists():
            return key_path.read_text().strip()
        key = os.urandom(32).hex()
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(key)
        try:
            os.chmod(key_path, 0o600)
        except OSError:
            pass
        return key
    except OSError:
        return os.urandom(32).hex()


# Security configuration
SECRET_KEY = _load_or_create_secret()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class SecurityManager:
    """Handle authentication and authorization."""
    
    def __init__(self):
        self.users_db = {}  # In production, use a proper database
        self._seed_default_user()

    def _seed_default_user(self):
        """Seed an initial admin user so login is possible out of the box."""
        username = os.getenv("JARVIS_ADMIN_USER", "admin")
        password = os.getenv("JARVIS_ADMIN_PASSWORD")
        if not password:
            password = "admin"
            logger.warning(
                "No JARVIS_ADMIN_PASSWORD set; seeding default 'admin'/'admin'. "
                "Set JARVIS_ADMIN_USER / JARVIS_ADMIN_PASSWORD before real use."
            )
        self.users_db[username] = SimpleNamespace(
            username=username,
            hashed_password=self.get_password_hash(password),
        )

    def authenticate_user(self, username: str, password: str):
        """Return the user object if credentials are valid, otherwise None."""
        user = self.users_db.get(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user

    def verify_token(self, token: Optional[str]) -> bool:
        """Return True if the JWT is valid (used for WebSocket auth)."""
        if not token:
            return False
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload.get("sub") is not None
        except JWTError:
            return False

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    async def get_current_user(self, token: str = Depends(oauth2_scheme)):
        """Validate and return current user."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
        except JWTError:
            raise credentials_exception
        user = self.users_db.get(username)
        if user is None:
            raise credentials_exception
        return user