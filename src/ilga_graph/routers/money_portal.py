"""Buyer-facing Illinois Influence money portal (/money).

Separate product shell from Land of Kei advocacy and the /intelligence/money
engine. Reuses campaign_finance view helpers and money_intel_leads only —
does not touch ingest, match, or GraphQL resolvers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from .. import config as cfg
from ..app_state import state
from ..db import get_db
from ..intelligence_helpers import (
    bill_money_context_view,
    campaign_finance_summary_view,
    is_sample_scale_finance,
    lookup_bill_record,
    lookup_member_query,
    lookup_member_record,
    member_money_trail_view,
    top_funded_member_rows,
)
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

LOGGER = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
router = APIRouter()
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
templates.env.globals["app_base_url"] = cfg.APP_BASE_URL
templates.env.globals["use_minified_assets"] = cfg.PROFILE == "prod"
templates.env.globals["umami_enabled"] = cfg.PROFILE == "prod" and bool(cfg.UMAMI_WEBSITE_ID)
templates.env.globals["umami_website_id"] = cfg.UMAMI_WEBSITE_ID
templates.env.globals["umami_script_url"] = cfg.UMAMI_SCRIPT_URL

PORTAL_NAME = "Illinois Influence"
SEED_MEMBER_ID = "3268"
SEED_MEMBER_NAME = "Don Harmon"
SEED_BILL = "SB0341"

_DO_NEXT = (
    {
        "verb": "Watch",
        "title": SEED_MEMBER_NAME,
        "body": "The money trail for a sitting member.",
        "href": f"/money/member/{SEED_MEMBER_ID}",
        "cta": "Open",
    },
    {
        "verb": "Ask",
        "title": SEED_BILL,
        "body": "Who funded the people on this bill. Not earmarked.",
        "href": f"/money/bill/{SEED_BILL}",
        "cta": "Open",
    },
    {
        "verb": "Flag",
        "title": "Note a match",
        "body": "When something needs a human look before you brief.",
        "href": f"/money/member/{SEED_MEMBER_ID}",
        "cta": "Open",
    },
)


def _client_ip(request: Request) -> str:
    """Return client IP from X-Forwarded-For or direct connection."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def _finance_summary() -> dict[str, Any] | None:
    return campaign_finance_summary_view(state.campaign_finance)


def _sample_scale() -> bool:
    return is_sample_scale_finance(_finance_summary())


def _money_trail_for_member(member: Any) -> dict[str, Any] | None:
    return member_money_trail_view(state.campaign_finance, member)


def _money_context_for_bill(bill: Any) -> dict[str, Any] | None:
    return bill_money_context_view(
        state.campaign_finance,
        bill,
        state.member_lookup_by_id,
        vote_events=state.vote_lookup.get(bill.bill_number, []),
    )


def _portal_shell(request: Request, title: str, **extra: Any) -> dict[str, Any]:
    """Shared Jinja context for every portal page."""
    return {
        "request": request,
        "title": title,
        "portal_name": PORTAL_NAME,
        "is_sample_scale": _sample_scale(),
        "seed_member_id": SEED_MEMBER_ID,
        "seed_member_name": SEED_MEMBER_NAME,
        "seed_bill": SEED_BILL,
        **extra,
    }


def _htmx_message(text: str, *, error: bool) -> HTMLResponse:
    """Return a status/alert fragment for HTMX form swap."""
    css = "mp-alert mp-alert--error" if error else "mp-alert mp-alert--ok"
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


@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
def money_landing(request: Request) -> Any:
    """Product landing: one promise, one email field."""
    status = request.query_params.get("status")
    return templates.TemplateResponse(
        request,
        "money_portal_landing.html",
        _portal_shell(request, PORTAL_NAME, **signup_form_context(status=status)),
    )


@router.get("/demo", include_in_schema=False)
def money_demo(request: Request, bill: str = "", member: str = "") -> Any:
    """Buyer demo: one member, one bill, Watch / Ask / Flag. Browse is below."""
    member_query = (member or "").strip()
    if member_query:
        found = lookup_member_query(member_query)
        if found:
            return RedirectResponse(f"/money/member/{found.id}", status_code=302)

    finance_summary = _finance_summary()
    funded_members = top_funded_member_rows(
        state.campaign_finance,
        state.member_lookup_by_id,
        limit=12,
    )
    bill_query = (bill or "").strip()
    bill_record = lookup_bill_record(bill_query) if bill_query else None
    bill_money = _money_context_for_bill(bill_record) if bill_record else None
    return templates.TemplateResponse(
        request,
        "money_portal_demo.html",
        _portal_shell(
            request,
            "Demo",
            finance_summary=finance_summary,
            funded_members=funded_members,
            bill_query=bill_query,
            bill_record=bill_record,
            bill_money=bill_money,
            bill_not_found=bool(bill_query) and bill_record is None,
            member_query=member_query,
            member_not_found=bool(member_query),
            do_next=_DO_NEXT,
        ),
    )


@router.get("/member/{member_id}", include_in_schema=False)
def money_member(request: Request, member_id: str) -> Any:
    """Member money trail in the buyer portal shell."""
    member = lookup_member_record(member_id) or lookup_member_query(member_id)
    money_trail = _money_trail_for_member(member) if member else None
    return templates.TemplateResponse(
        request,
        "money_portal_member.html",
        _portal_shell(
            request,
            member.name if member else "Member",
            member=member,
            money_trail=money_trail,
            do_next=_DO_NEXT,
        ),
    )


@router.get("/bill/{bill}", include_in_schema=False)
def money_bill(request: Request, bill: str) -> Any:
    """Bill money context in the buyer portal shell."""
    bill_record = lookup_bill_record(bill)
    bill_money = _money_context_for_bill(bill_record) if bill_record else None
    return templates.TemplateResponse(
        request,
        "money_portal_bill.html",
        _portal_shell(
            request,
            bill_record.bill_number if bill_record else bill,
            bill_query=bill,
            bill_record=bill_record,
            bill_money=bill_money,
            do_next=_DO_NEXT,
        ),
    )


@router.get("/signup", include_in_schema=False)
def money_signup_page(request: Request) -> Any:
    """Canonical one-field access form (same store as the landing box)."""
    status = request.query_params.get("status")
    return templates.TemplateResponse(
        request,
        "money_portal_signup.html",
        _portal_shell(
            request,
            "Waitlist",
            **signup_form_context(status=status),
        ),
    )


@router.post("/signup", include_in_schema=False)
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
        return RedirectResponse("/money/signup?status=csrf", status_code=303)

    if not rate_limit_money_lead(_client_ip(request)):
        if is_htmx:
            return _htmx_message(STATUS_MESSAGES["rate"], error=True)
        return RedirectResponse("/money/signup?status=rate", status_code=303)

    form = await request.form()
    raw_roles = [str(v) for v in form.getlist("role")]
    normalized = normalize_email(email)
    if not normalized:
        if is_htmx:
            return templates.TemplateResponse(
                request,
                "_money_portal_signup_form.html",
                {
                    **_portal_shell(
                        request,
                        "Waitlist",
                        **signup_form_context(
                            status="invalid",
                            form_values={
                                "email": email,
                                "name": name or "",
                                "org": org or "",
                                "roles": [r for r in raw_roles if r in MONEY_LEAD_ROLES],
                            },
                        ),
                    ),
                },
                status_code=400,
            )
        return RedirectResponse("/money/signup?status=invalid", status_code=303)

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
    return RedirectResponse(f"/money/signup?status={status}", status_code=303)
