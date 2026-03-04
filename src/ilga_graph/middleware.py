"""Register CORS, API key, CSRF, security headers, and request logging on the FastAPI app."""

from __future__ import annotations

import logging
import sys
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy import select
from starlette.middleware.gzip import GZipMiddleware

from . import config as cfg
from .campaign_helpers import get_active_campaign
from .db import async_session_factory
from .db_models import User
from .dependencies import decode_session_token
from .security import (
    CSRF_COOKIE_NAME,
    CSRF_MAX_AGE_SECONDS,
    generate_csrf_token,
    is_valid_csrf_token,
)

LOGGER = logging.getLogger(__name__)


def _build_csp_directive() -> str:
    """Build CSP directive: self + trusted CDNs (HTMX, Tippy, Umami, Turnstile, D3)."""
    script_src = (
        "'self' 'unsafe-inline' https://unpkg.com https://cdn.jsdelivr.net "
        "https://cloud.umami.is https://challenges.cloudflare.com https://d3js.org"
    )
    parts = [
        "default-src 'self'",
        f"script-src {script_src}",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: https:",
        "connect-src 'self' https://cloud.umami.is https://challenges.cloudflare.com",
        "frame-src https://challenges.cloudflare.com",
    ]
    directive = "; ".join(parts)
    if cfg.CSP_REPORT_URI:
        directive += f"; report-uri {cfg.CSP_REPORT_URI}"
    return directive


def register_middleware(app: FastAPI) -> None:
    """Add CORS, API key auth, CSRF cookie, security headers, and request logging."""
    app.add_middleware(GZipMiddleware, minimum_size=500)
    origins = [o.strip() for o in cfg.CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def current_action_campaign_middleware(request: Request, call_next) -> Response:
        """Set request.state.current_action_campaign and request.state.user for HTML requests."""
        accept = request.headers.get("accept") or ""
        if "text/html" in accept and not request.url.path.startswith("/static"):
            try:
                async with async_session_factory() as db:
                    campaign = await get_active_campaign(db)
                    request.state.current_action_campaign = campaign  # type: ignore[attr-defined]
                    session_cookie = request.cookies.get(cfg.AUTH_COOKIE_NAME)
                    user_id = decode_session_token(session_cookie) if session_cookie else None
                    if user_id is not None:
                        result = await db.execute(select(User).where(User.id == user_id))
                        user = result.scalar_one_or_none()
                    else:
                        user = None
                    request.state.user = user  # type: ignore[attr-defined]  # templates hide subscribe when user.wants_updates
            except Exception:
                request.state.current_action_campaign = None  # type: ignore[attr-defined]
                request.state.user = None  # type: ignore[attr-defined]
        return await call_next(request)

    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next) -> Response:
        if cfg.API_KEY:
            exempt = {
                "/",
                "/health",
                "/docs",
                "/openapi.json",
                "/redoc",
                "/favicon.ico",
                "/sitemap.xml",
                "/robots.txt",
                "/privacy",
                "/terms",
            }
            path = request.url.path
            accept = request.headers.get("accept") or ""
            # Let browser GETs through so unknown paths get the custom 404 page, not 401.
            browser_get = request.method == "GET" and "text/html" in accept
            if (
                not browser_get
                and path not in exempt
                and not path.startswith("/admin")
                and not path.startswith("/advocacy")
                and not path.startswith("/auth")
                and not path.startswith("/outreach")
                and not path.startswith("/report-bug")
                and not path.startswith("/explore")
                and not path.startswith("/intelligence")
                and not path.startswith("/api/graph")
                and not path.startswith("/api/dev")
                and not path.startswith("/dev")
                and not path.startswith("/static")
                and not path.startswith("/updates")
                and request.method != "OPTIONS"
            ):
                provided = request.headers.get("X-API-Key", "")
                if provided != cfg.API_KEY:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing API key"},
                    )
        return await call_next(request)

    @app.middleware("http")
    async def csrf_cookie_middleware(request: Request, call_next) -> Response:
        existing = request.cookies.get(CSRF_COOKIE_NAME)
        token = existing if existing and is_valid_csrf_token(existing) else generate_csrf_token()
        request.state.csrf_token = token  # type: ignore[attr-defined]
        response: Response = await call_next(request)
        if hasattr(response, "set_cookie"):
            response.set_cookie(
                key=CSRF_COOKIE_NAME,
                value=token,
                max_age=CSRF_MAX_AGE_SECONDS,
                path="/",
                httponly=False,
                samesite="strict",
                secure=request.url.scheme == "https",
            )
        return response

    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next) -> Response:
        response: Response = await call_next(request)
        if hasattr(response, "headers"):
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
            csp = _build_csp_directive()
            if cfg.CSP_ENFORCE:
                response.headers["Content-Security-Policy"] = csp
            else:
                response.headers["Content-Security-Policy-Report-Only"] = csp
            if cfg.PROFILE == "prod" and cfg.HSTS_ENABLED:
                response.headers["Strict-Transport-Security"] = (
                    "max-age=31536000; includeSubDomains"
                )
        return response

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next) -> Response:
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        line = f"{request.method} {request.url.path} {response.status_code} ({elapsed_ms:.1f}ms)"
        LOGGER.info(
            "%s %s %d (%.1fms)", request.method, request.url.path, response.status_code, elapsed_ms
        )
        # In dev, echo each request to stderr so traffic is visible in the terminal.
        if cfg.DEV_MODE:
            print(line, file=sys.stderr, flush=True)
        return response
