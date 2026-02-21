"""Site-level routes: root redirect, advocacy redirect, favicon, sitemap, robots."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse, Response

from .. import config as cfg

router = APIRouter()
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_SITEMAP_PATHS = ("/", "/advocacy", "/intelligence/", "/explore", "/privacy", "/terms")


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect root to the advocacy page."""
    return RedirectResponse(url="/advocacy", status_code=302)


@router.get("/advocacy", include_in_schema=False)
def advocacy_trailing_slash_redirect() -> RedirectResponse:
    """Ensure /advocacy is served: mounted router receives path ''; redirect so child sees '/'."""
    return RedirectResponse(url="/advocacy/", status_code=302)


@router.get("/intelligence", include_in_schema=False)
def intelligence_trailing_slash_redirect() -> RedirectResponse:
    """Redirect /intelligence → /intelligence/ so sitemap/bookmarks don't 404."""
    return RedirectResponse(url="/intelligence/", status_code=302)


@router.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    """Serve theme-matching favicon (Kei truck SVG) at /favicon.ico."""
    path = _STATIC_DIR / "favicon.svg"
    return FileResponse(path, media_type="image/svg+xml")


@router.get("/sitemap.xml", include_in_schema=False)
def sitemap_xml() -> Response:
    """Serve sitemap.xml for search engine discovery; URLs use APP_BASE_URL."""
    base = cfg.APP_BASE_URL
    urls = "".join(f"    <url><loc>{base}{path}</loc></url>\n" for path in _SITEMAP_PATHS)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}"
        "</urlset>\n"
    )
    return Response(content=xml, media_type="application/xml; charset=utf-8")


@router.get("/robots.txt", include_in_schema=False)
def robots_txt() -> PlainTextResponse:
    """Serve robots.txt (allow all, point to sitemap)."""
    body = f"User-agent: *\nAllow: /\nSitemap: {cfg.APP_BASE_URL}/sitemap.xml\n"
    return PlainTextResponse(content=body)
