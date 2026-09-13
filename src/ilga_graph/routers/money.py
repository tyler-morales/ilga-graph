"""Shareable money-intel waitlist (POST + GET /money/signup).

GET /intelligence/money stays on the Follow-the-money engine in intelligence.py.
This router does not claim that path.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..app_state import state
from ..db import get_db
from ..intelligence_helpers import campaign_finance_summary_view, is_sample_scale_finance
from ..money_leads import (
    MONEY_LEAD_ROLES,
    STATUS_MESSAGES,
    normalize_email,
    normalize_name,
    normalize_org,
    normalize_roles,
    persist_money_lead,
    signup_form_context,
)
from ..security import CSRF_COOKIE_NAME, rate_limit_money_lead, validate_csrf_token
from .intelligence import templates

LOGGER = logging.getLogger(__name__)

router = APIRouter()


def _client_ip(request: Request) -> str:
    """Return client IP from X-Forwarded-For or direct connection."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def _signup_page_context(
    request: Request,
    *,
    status: str | None = None,
    form_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Context for the shareable signup-only page."""
    return {
        "request": request,
        "title": "Money intel signup",
        "is_sample_scale_finance": is_sample_scale_finance(
            campaign_finance_summary_view(state.campaign_finance)
        ),
        **signup_form_context(status=status, form_values=form_values),
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


@router.get("/money/signup", include_in_schema=False)
def money_signup_page(request: Request) -> Any:
    """Dedicated short signup page (shareable URL)."""
    status = request.query_params.get("status")
    return templates.TemplateResponse(
        request,
        "intelligence_money_signup.html",
        _signup_page_context(request, status=status),
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
            return _htmx_message(STATUS_MESSAGES["csrf"], error=True)
        return RedirectResponse("/intelligence/money/signup?status=csrf", status_code=303)

    if not rate_limit_money_lead(_client_ip(request)):
        if is_htmx:
            return _htmx_message(STATUS_MESSAGES["rate"], error=True)
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
                    **_signup_page_context(
                        request,
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
        key = "already" if result == "already" else "ok"
        return _htmx_message(STATUS_MESSAGES[key], error=False)
    status = "already" if result == "already" else "ok"
    return RedirectResponse(f"/intelligence/money/signup?status={status}", status_code=303)
