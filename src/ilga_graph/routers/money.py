"""Legacy /intelligence/money/signup → canonical /money/signup.

GET /intelligence/money stays on the Follow-the-money engine in intelligence.py.
This router only redirects the old shareable waitlist URL.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

router = APIRouter()


def _signup_redirect(request: Request, *, status_code: int) -> RedirectResponse:
    """Send buyers to the portal waitlist, preserving query string."""
    query = request.url.query
    location = "/money/signup" + (f"?{query}" if query else "")
    return RedirectResponse(location, status_code=status_code)


@router.get("/money/signup", include_in_schema=False)
def money_signup_page(request: Request) -> RedirectResponse:
    """Canonical buyer URL is /money/signup."""
    return _signup_redirect(request, status_code=302)


@router.post("/money/signup", include_in_schema=False)
async def money_signup_post(request: Request) -> RedirectResponse:
    """Preserve method/body so old form posts still land on the portal handler."""
    return _signup_redirect(request, status_code=307)
