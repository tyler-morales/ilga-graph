"""Money-intel landing and lobbyist/buyer email waitlist (v1.1 lead capture).

Lives on /intelligence/money. Not SOS/lobbyist disclosure join; not Moneyball.
Single-opt: email is stored immediately. Export via /admin/money-leads.csv.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..money_leads import (
    MONEY_LEAD_ROLES,
    normalize_email,
    normalize_name,
    normalize_org,
    normalize_roles,
    persist_money_lead,
)
from ..security import CSRF_COOKIE_NAME, rate_limit_money_lead, validate_csrf_token
from .intelligence import templates

LOGGER = logging.getLogger(__name__)

router = APIRouter()

# Demo seeds for the live money-intel surface (member trail / bill sponsor context).
_DEMO_MEMBER_ID = "3268"
_DEMO_MEMBER_NAME = "Don Harmon"
_DEMO_BILL_NUMBER = "SB0341"

_STATUS_MESSAGES = {
    "ok": "You're on the list. We'll email when follow-the-money intel expands.",
    "already": "You're already on the list. We'll keep you posted.",
    "invalid": "Please enter a valid email address.",
    "csrf": "Invalid or expired security token. Reload the page and try again.",
    "rate": "Too many signup attempts. Try again later.",
}


def _client_ip(request: Request) -> str:
    """Return client IP from X-Forwarded-For or direct connection."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def _page_context(
    request: Request,
    *,
    dedicated: bool,
    status: str | None = None,
    form_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared template context for landing and dedicated signup pages."""
    values = form_values or {}
    return {
        "request": request,
        "title": "Money intel" if not dedicated else "Money intel signup",
        "dedicated": dedicated,
        "status": status or "",
        "status_message": _STATUS_MESSAGES.get(status or "", ""),
        "status_is_error": status in ("invalid", "csrf", "rate"),
        "demo_member_id": _DEMO_MEMBER_ID,
        "demo_member_name": _DEMO_MEMBER_NAME,
        "demo_bill_number": _DEMO_BILL_NUMBER,
        "role_choices": (
            ("lobbyist", "Lobbyist"),
            ("lawyer", "Lawyer"),
            ("nonprofit", "Nonprofit"),
        ),
        "form_email": values.get("email", ""),
        "form_name": values.get("name", ""),
        "form_org": values.get("org", ""),
        "form_roles": values.get("roles") or [],
    }


def _htmx_message(text: str, *, error: bool) -> HTMLResponse:
    """Return a status/alert fragment for HTMX form swap."""
    css = "subscribe-email-error" if error else "subscribe-email-success"
    role = "alert" if error else "status"
    code = 400 if error else 200
    if "Too many" in text:
        code = 429
    if "security token" in text.lower():
        code = 403
    return HTMLResponse(
        f'<p class="{css}" role="{role}">{text}</p>',
        status_code=code,
    )


@router.get("/money", include_in_schema=False)
def money_landing(request: Request) -> Any:
    """Money intel surface: value prop, demo seeds, and waitlist form."""
    status = request.query_params.get("status")
    return templates.TemplateResponse(
        request,
        "intelligence_money.html",
        _page_context(request, dedicated=False, status=status),
    )


@router.get("/money/signup", include_in_schema=False)
def money_signup_page(request: Request) -> Any:
    """Dedicated short signup page (shareable URL)."""
    status = request.query_params.get("status")
    return templates.TemplateResponse(
        request,
        "intelligence_money.html",
        _page_context(request, dedicated=True, status=status),
    )


@router.post("/money/signup", include_in_schema=False)
async def money_signup_post(
    request: Request,
    email: str = Form(..., max_length=320),
    name: str | None = Form(None, max_length=120),
    org: str | None = Form(None, max_length=200),
    csrf_token: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Single-opt waitlist: persist email (+ optional name, org, role)."""
    is_htmx = bool(request.headers.get("HX-Request"))
    token = csrf_token or request.headers.get("X-XSRF-TOKEN")
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not validate_csrf_token(token, cookie_token):
        if is_htmx:
            return _htmx_message(_STATUS_MESSAGES["csrf"], error=True)
        return RedirectResponse("/intelligence/money/signup?status=csrf", status_code=303)

    if not rate_limit_money_lead(_client_ip(request)):
        if is_htmx:
            return _htmx_message(_STATUS_MESSAGES["rate"], error=True)
        return RedirectResponse("/intelligence/money/signup?status=rate", status_code=303)

    form = await request.form()
    raw_roles = [str(v) for v in form.getlist("role")]
    normalized = normalize_email(email)
    if not normalized:
        if is_htmx:
            return templates.TemplateResponse(
                request,
                "_money_signup_form.html",
                {
                    **_page_context(
                        request,
                        dedicated=True,
                        status="invalid",
                        form_values={
                            "email": email,
                            "name": name or "",
                            "org": org or "",
                            "roles": [r for r in raw_roles if r in MONEY_LEAD_ROLES],
                        },
                    ),
                },
                status_code=400,
            )
        return RedirectResponse("/intelligence/money/signup?status=invalid", status_code=303)

    result = await persist_money_lead(
        db,
        email=normalized,
        name=normalize_name(name),
        org=normalize_org(org),
        role=normalize_roles(raw_roles),
    )
    LOGGER.info("Money intel lead %s: email=%s", result, normalized)
    if is_htmx:
        return _htmx_message(
            _STATUS_MESSAGES["already" if result == "already" else "ok"], error=False
        )
    status = "already" if result == "already" else "ok"
    return RedirectResponse(f"/intelligence/money/signup?status={status}", status_code=303)
