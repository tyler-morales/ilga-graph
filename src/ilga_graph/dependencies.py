"""FastAPI dependencies for auth and database sessions."""

from __future__ import annotations

import logging

from fastapi import Cookie, Depends
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import config as cfg
from .db import get_db
from .db_models import User

LOGGER = logging.getLogger(__name__)

_signer = URLSafeTimedSerializer(cfg.AUTH_SECRET)


def create_session_token(user_id: int) -> str:
    """Create a signed, time-limited session token for the given user."""
    return _signer.dumps({"uid": user_id})


def decode_session_token(token: str) -> int | None:
    """Decode a session token, returning user_id or None if invalid/expired."""
    try:
        data = _signer.loads(token, max_age=cfg.AUTH_COOKIE_MAX_AGE)
        return data.get("uid")
    except (BadSignature, SignatureExpired):
        return None


async def get_current_user_optional(
    db: AsyncSession = Depends(get_db),
    ilga_session: str | None = Cookie(None, alias=cfg.AUTH_COOKIE_NAME),
) -> User | None:
    """Return the logged-in User or None.  Never raises — anonymous is OK."""
    if not ilga_session:
        return None
    user_id = decode_session_token(ilga_session)
    if user_id is None:
        return None
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def require_user(
    user: User | None = Depends(get_current_user_optional),
) -> User:
    """Raise 401 if the user is not authenticated."""
    if user is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
