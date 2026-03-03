"""Content pages: The Issue and Legislator Brief."""

from __future__ import annotations

import html as html_module
import json
import re
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import config as cfg
from ..campaign_config import get_campaign_config
from ..constants import KEI_POLL_IMPACT_OPTIONS, KEI_STATUS_OPTIONS
from ..db import get_db
from ..db_models import CommunityStory, KeiInterestStatement, User
from ..dependencies import get_current_user_optional
from ..session_schedule import (
    get_all_deadlines,
    get_milestone_by_id,
    get_next_deadline_safe,
    session_label,
)
from .content_constants import (
    BRIEF_BILLS_CURRENT,
    BRIEF_BILLS_PASSED,
    BRIEF_DOCUMENTS,
    BRIEF_SOURCES,
    BRIEF_STATE_STATUS,
    CAMPAIGN_STATUS,
    CONSTITUENT_BRIEF_PATH,
    DOMAIN_GLOSSARY,
    FACT_SHEET_ISSUE,
    FACT_SHEET_PDF_URL,
    FACT_SHEET_POSITION,
    FAQ_ADVOCACY,
    FAQ_LAW,
    FAQ_LEGISLATORS,
    FAQ_SESSION,
    HERO_CLARITY_LINE,
    HERO_URGENCY_LINE,
    HERO_URGENCY_THIS_SESSION,
    INTRO_CARD_WHY_CALL,
    ISSUE_SOURCES,
    KEI_GLOSSARY,
    KEI_POLL_WHY_WE_ASK,
    KEI_POLL_WIDE_NET_LINE,
    LEGISLATOR_BRIEF_PATH,
    MARQUEE_IMAGES,
    PROGRESS_ACHIEVED_COUNT,
    PROGRESS_CHECKPOINTS,
    SESSION_SCHEDULE_TERMS,
    STRATEGIC_FIVE_POINTS,
    STRATEGIC_MISSION,
    STRATEGIC_STATES_ICON_ABBR,
    STRATEGIC_VISION,
    TIMELINE_PHASES,
    WHY_SHOULD_YOU_CARE_HEADING,
    WHY_SHOULD_YOU_CARE_INTRO,
    WHY_SHOULD_YOU_CARE_TEASER_HEADING,
    WHY_SHOULD_YOU_CARE_TEASER_ITEMS,
    WHY_SHOULD_YOU_CARE_VOICE,
    WHY_YOU_CARE_BRANCHES,
    WHY_YOU_CARE_CTA_NUDGE,
    WHY_YOU_CARE_DEFAULT_CARDS,
    WHY_YOU_CARE_PRE_POLL_LINE,
)

# Re-exports for advocacy, home, updates, etc.
__all__ = [
    "BRIEF_BILLS_CURRENT",
    "BRIEF_BILLS_PASSED",
    "BRIEF_DOCUMENTS",
    "BRIEF_SOURCES",
    "BRIEF_STATE_STATUS",
    "CAMPAIGN_STATUS",
    "CONSTITUENT_BRIEF_PATH",
    "DOMAIN_GLOSSARY",
    "FACT_SHEET_ISSUE",
    "FACT_SHEET_PDF_URL",
    "FACT_SHEET_POSITION",
    "FAQ_ADVOCACY",
    "FAQ_LAW",
    "FAQ_LEGISLATORS",
    "FAQ_SESSION",
    "HERO_CLARITY_LINE",
    "HERO_URGENCY_LINE",
    "HERO_URGENCY_THIS_SESSION",
    "INTRO_CARD_WHY_CALL",
    "ISSUE_SOURCES",
    "KEI_GLOSSARY",
    "KEI_POLL_WHY_WE_ASK",
    "KEI_POLL_WIDE_NET_LINE",
    "LEGISLATOR_BRIEF_PATH",
    "MARQUEE_IMAGES",
    "PROGRESS_ACHIEVED_COUNT",
    "PROGRESS_CHECKPOINTS",
    "SESSION_SCHEDULE_TERMS",
    "STRATEGIC_FIVE_POINTS",
    "STRATEGIC_MISSION",
    "STRATEGIC_STATES_ICON_ABBR",
    "STRATEGIC_VISION",
    "TIMELINE_PHASES",
    "WHY_SHOULD_YOU_CARE_HEADING",
    "WHY_SHOULD_YOU_CARE_INTRO",
    "WHY_SHOULD_YOU_CARE_TEASER_HEADING",
    "WHY_SHOULD_YOU_CARE_TEASER_ITEMS",
    "WHY_SHOULD_YOU_CARE_VOICE",
    "WHY_YOU_CARE_BRANCHES",
    "WHY_YOU_CARE_DEFAULT_CARDS",
    "WHY_YOU_CARE_PRE_POLL_LINE",
    "WHY_YOU_CARE_CTA_NUDGE",
]

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
router = APIRouter()


def _load_constituent_brief() -> dict | None:
    """Parse constituent brief .txt into title, subtitle, and sections. Returns None if file missing."""
    if not CONSTITUENT_BRIEF_PATH.is_file():
        return None
    text = CONSTITUENT_BRIEF_PATH.read_text(encoding="utf-8", errors="replace").strip()
    parts = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if len(parts) < 2:
        return None
    title = parts[0]
    subtitle = parts[1] if len(parts) > 1 else ""
    sections: list[dict] = []
    i = 2
    while i + 1 < len(parts):
        heading = parts[i]
        body = parts[i + 1]
        bullets: list[str] = []
        paragraphs: list[str] = []
        for line in body.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Bullet: starts with • (U+2022) or "- " / "• "
            if line.startswith("•") or re.match(r"^[-•]\s*", line):
                bullets.append(line.lstrip("••-	 ").strip())
            else:
                paragraphs.append(line)
        section: dict = {"heading": heading, "paragraphs": paragraphs}
        if bullets:
            section["bullets"] = bullets
        sections.append(section)
        i += 2
    return {"title": title, "subtitle": subtitle, "sections": sections}


def _load_legislator_brief() -> dict | None:
    """Parse legislator brief .txt into structured fields. Returns None if file missing."""
    if not LEGISLATOR_BRIEF_PATH.is_file():
        return None
    text = LEGISLATOR_BRIEF_PATH.read_text(encoding="utf-8", errors="replace").strip()
    lines = text.split("\n")
    out: dict = {
        "title": "",
        "subtitle": "",
        "issue_one_sentence": "",
        "core_ambiguity": "",
        "sections": [],
        "ask_list": [],
        "attachments": "",
        "statutory_ref": "",
        "point_of_contact": "",
    }
    i = 0
    if i < len(lines):
        out["title"] = lines[i].strip()
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines):
        out["subtitle"] = lines[i].strip()
        i += 1
    while i < len(lines):
        line = lines[i]
        if line.startswith("Issue in one sentence:"):
            out["issue_one_sentence"] = line.split(":", 1)[1].strip()
            i += 1
            continue
        if line.startswith("Core ambiguity:"):
            out["core_ambiguity"] = line.split(":", 1)[1].strip()
            i += 1
            continue
        if line.startswith("Illinois statutory reference:"):
            out["statutory_ref"] = line.split(":", 1)[1].strip()
            i += 1
            continue
        if line.startswith("Point of contact:"):
            out["point_of_contact"] = line.split(":", 1)[1].strip()
            i += 1
            continue
        if line.strip() == "Attachments":
            i += 1
            attach_lines = []
            while (
                i < len(lines)
                and lines[i].strip()
                and not lines[i].startswith("Illinois ")
                and not lines[i].startswith("Point of ")
            ):
                attach_lines.append(lines[i].strip())
                i += 1
            out["attachments"] = "\n".join(attach_lines)
            continue
        if line.strip() == "What we are asking your office to do":
            i += 1
            ask_items = []
            while i < len(lines) and lines[i].strip():
                rest = re.sub(r"^\d\)\s*", "", lines[i].strip())
                if rest:
                    ask_items.append(rest)
                i += 1
            out["ask_list"] = ask_items
            continue
        # Section heading (title case, no colon at end)
        stripped = line.strip()
        if (
            stripped
            and i + 1 < len(lines)
            and not stripped.startswith(("Issue in", "Core ambiguity", "Illinois ", "Point of "))
        ):
            section_body: list[str] = []
            i += 1
            while (
                i < len(lines)
                and lines[i].strip()
                and not re.match(
                    r"^(What we are asking|Attachments|Illinois statutory|Point of contact)",
                    lines[i],
                )
            ):
                section_body.append(lines[i].strip())
                i += 1
            if section_body:
                out["sections"].append({"heading": stripped, "paragraphs": section_body})
            continue
        i += 1
    return out


templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
templates.env.globals["dev_available"] = cfg.DEV_MODE
templates.env.globals["app_base_url"] = cfg.APP_BASE_URL
templates.env.globals["site_name"] = cfg.SITE_NAME
_campaign = get_campaign_config()
templates.env.globals["campaign_name"] = _campaign.campaign_name or cfg.SITE_NAME
templates.env.globals["primary_color"] = _campaign.primary_color or "#FF4500"
templates.env.globals["issue_summary"] = _campaign.issue_summary
templates.env.globals["meta_description"] = cfg.META_DESCRIPTION
templates.env.globals["og_image_url"] = cfg.OG_IMAGE_URL
templates.env.globals["umami_enabled"] = cfg.PROFILE == "prod" and bool(cfg.UMAMI_WEBSITE_ID)
templates.env.globals["umami_website_id"] = cfg.UMAMI_WEBSITE_ID
templates.env.globals["umami_script_url"] = cfg.UMAMI_SCRIPT_URL
templates.env.globals["show_beta_banner"] = cfg.BETA_BANNER
templates.env.globals["beta_banner_feedback_url"] = cfg.BETA_BANNER_REPORT_URL
templates.env.globals["footer_last_updated"] = cfg.FOOTER_LAST_UPDATED
templates.env.globals["footer_last_updated_iso"] = cfg.FOOTER_LAST_UPDATED_ISO
templates.env.globals["strategic_five_points"] = STRATEGIC_FIVE_POINTS
templates.env.globals["features"] = cfg.get_client_features()
templates.env.globals["marquee_images"] = MARQUEE_IMAGES  # Overridden per-request when db available
templates.env.globals["why_should_you_care_heading"] = WHY_SHOULD_YOU_CARE_HEADING
templates.env.globals["why_should_you_care_teaser_heading"] = WHY_SHOULD_YOU_CARE_TEASER_HEADING
templates.env.globals["why_should_you_care_intro"] = WHY_SHOULD_YOU_CARE_INTRO
templates.env.globals["why_should_you_care_voice"] = WHY_SHOULD_YOU_CARE_VOICE
templates.env.globals["why_should_you_care_teaser_items"] = WHY_SHOULD_YOU_CARE_TEASER_ITEMS
templates.env.globals["why_you_care_default_cards"] = WHY_YOU_CARE_DEFAULT_CARDS
templates.env.globals["why_you_care_pre_poll_line"] = WHY_YOU_CARE_PRE_POLL_LINE
templates.env.globals["why_you_care_cta_nudge"] = WHY_YOU_CARE_CTA_NUDGE
templates.env.globals["why_you_care_branches"] = WHY_YOU_CARE_BRANCHES
templates.env.globals["kei_poll_why_we_ask"] = KEI_POLL_WHY_WE_ASK
templates.env.globals["kei_poll_wide_net_line"] = KEI_POLL_WIDE_NET_LINE

from ..campaign_helpers import get_current_action_campaign_for_template  # noqa: E402

templates.env.globals["get_current_action_campaign"] = get_current_action_campaign_for_template
templates.env.globals["get_milestone_by_id"] = get_milestone_by_id
templates.env.globals["get_next_deadline"] = get_next_deadline_safe
templates.env.globals["kei_status_options"] = KEI_STATUS_OPTIONS
templates.env.globals["kei_impact_options"] = KEI_POLL_IMPACT_OPTIONS
templates.env.globals["turnstile_site_key"] = (
    "" if cfg.TURNSTILE_DISABLED else (cfg.TURNSTILE_SITE_KEY or "")
)


async def get_marquee_images(db: AsyncSession) -> list[dict[str, str]]:
    """Return MARQUEE_IMAGES plus approved community story submissions (image-only). Legacy; prefer get_marquee_items."""
    items = await get_marquee_items(db)
    return [x for x in items if x.get("type") == "image" and "src" in x]


def _story_image_path_exists(image_path: str) -> bool:
    """True if the story image file exists under static (avoids showing broken images in marquee)."""
    if not image_path:
        return False
    static_dir = Path(__file__).resolve().parent.parent / "static"
    return (static_dir / image_path).exists()


async def get_marquee_items(db: AsyncSession) -> list[dict]:
    """Return unified marquee items: type=image (MARQUEE_IMAGES + approved CommunityStory) and type=text (approved KeiInterestStatement)."""
    image_items: list[dict] = [
        {
            "type": "image",
            "src": m["src"],
            "alt": m.get("alt", ""),
            "name": m.get("name", ""),
            "caption": m.get("caption", ""),
            "location": m.get("location", ""),
        }
        for m in MARQUEE_IMAGES
    ]
    result = await db.execute(
        select(CommunityStory)
        .where(CommunityStory.status == "approved")
        .order_by(CommunityStory.reviewed_at)
    )
    for s in result.scalars().all():
        if not _story_image_path_exists(s.image_path):
            continue
        image_items.append(
            {
                "type": "image",
                "src": f"/static/{s.image_path}",
                "alt": f"{s.name}, kei vehicle owner",
                "name": s.name,
                "caption": s.story,
                "location": s.location or "",
            }
        )
    result = await db.execute(
        select(KeiInterestStatement)
        .where(KeiInterestStatement.status == "approved")
        .order_by(KeiInterestStatement.reviewed_at)
    )
    text_items: list[dict] = [
        {
            "type": "text",
            "name": s.name,
            "statement": s.statement,
            "location": s.location or "",
        }
        for s in result.scalars().all()
    ]
    return image_items + text_items


def get_strategic_states_tooltips() -> dict[str, dict[str, str]]:
    """Tooltip data for states in the strategic states icon: state name and law/policy (from BRIEF_STATE_STATUS)."""
    return {
        s["state_abbr"]: {"state": s["state"], "how": s.get("how") or "Allows registration"}
        for s in BRIEF_STATE_STATUS
        if s["state_abbr"] in STRATEGIC_STATES_ICON_ABBR
    }


def _brief_map_fill_status(s: dict) -> str:
    """Derive map fill key from BRIEF_STATE_STATUS row: passed | pending | policy | restricted | none."""
    if s.get("restricted"):
        return "restricted"
    if s["bill_status"] in ("passed", "pending"):
        return s["bill_status"]
    if s.get("policy"):
        return "policy"
    return "none"


def _brief_aamva_fix_state_abbrs() -> list[str]:
    """State abbrs where aamva_fix is True (reversed prior ban or passed explicit kei law)."""
    return [s["state_abbr"] for s in BRIEF_STATE_STATUS if s.get("aamva_fix")]


def _inline_glossary_terms(include_domain: bool = False) -> list[dict]:
    """Merge KEI + SESSION terms (and optionally DOMAIN) for inline tooltips. Sorted by term length descending for correct match order."""
    terms = list(KEI_GLOSSARY) + list(SESSION_SCHEDULE_TERMS)
    if include_domain:
        terms = terms + list(DOMAIN_GLOSSARY)
    return sorted(terms, key=lambda d: len(d["term"]), reverse=True)


def apply_inline_glossary(blocks: list[str], terms: list[dict]) -> list[str]:
    """Replace first occurrence of each term in blocks (document order) with a button+popover snippet. Returns HTML strings."""
    used_ids: set[str] = set()
    result: list[str] = []
    replacement_counter = 0

    for block in blocks:
        if not block or not block.strip():
            result.append(block)
            continue
        text = block
        changed = True
        while changed:
            changed = False
            best_pos = -1
            best_term: dict | None = None
            best_original_slice = ""

            for t in terms:
                if t["id"] in used_ids:
                    continue
                term = t["term"]
                pos = text.lower().find(term.lower())
                if pos < 0:
                    continue
                if best_pos < 0 or pos < best_pos:
                    best_pos = pos
                    best_term = t
                    best_original_slice = text[pos : pos + len(term)]

            if best_term is None or best_pos < 0:
                break
            replacement_counter += 1
            uid = f"{best_term['id']}-{replacement_counter}"
            def_escaped = html_module.escape(best_term["definition"])
            snippet = (
                f'<span class="tooltip-wrap tooltip-click">'
                f'<button type="button" class="glossary-inline-term" aria-expanded="false" aria-controls="glossary-def-{uid}" id="glossary-trigger-{uid}">'
                f"{html_module.escape(best_original_slice)}</button>"
                f'<span class="tooltip-content glossary-inline-def" role="tooltip" id="glossary-def-{uid}" hidden>'
                f'{def_escaped} <a href="/glossary#glossary-{best_term["id"]}">Full glossary</a></span></span>'
            )
            text = text[:best_pos] + snippet + text[best_pos + len(best_original_slice) :]
            used_ids.add(best_term["id"])
            changed = True

        result.append(text)

    return result


def _current_timeline_phase_id(today: date | None = None) -> str:
    """Return the timeline phase id that contains *today* (default: today). Before first phase → first; after last → last."""
    d = today or date.today()
    for phase in TIMELINE_PHASES:
        start = phase.get("start_date")
        end = phase.get("end_date")
        if start and end:
            if start <= d.isoformat() <= end:
                return phase["id"]
    if TIMELINE_PHASES:
        first_start = TIMELINE_PHASES[0].get("start_date")
        last_end = TIMELINE_PHASES[-1].get("end_date")
        if first_start and d.isoformat() < first_start:
            return TIMELINE_PHASES[0]["id"]
        if last_end and d.isoformat() > last_end:
            return TIMELINE_PHASES[-1]["id"]
    return TIMELINE_PHASES[0]["id"] if TIMELINE_PHASES else "build"


def _timeline_waterfall_data(timeline_phases: list[dict]) -> dict:
    """Build months list and per-phase start_col/end_col for waterfall table. Returns enriched phase copies."""
    if not timeline_phases:
        return {"months": [], "phases": []}
    starts = [p.get("start_date") for p in timeline_phases if p.get("start_date")]
    ends = [p.get("end_date") for p in timeline_phases if p.get("end_date")]
    if not starts or not ends:
        return {"months": [], "phases": []}
    first_ym = min(s[:7] for s in starts)
    last_ym = max(e[:7] for e in ends)
    y, m = int(first_ym[:4]), int(first_ym[5:7])
    ey, em = int(last_ym[:4]), int(last_ym[5:7])
    months: list[dict] = []
    while (y, m) <= (ey, em):
        months.append({"key": f"{y}-{m:02d}", "label": date(y, m, 1).strftime("%b %Y")})
        m += 1
        if m > 12:
            m, y = 1, y + 1
    month_keys = [mo["key"] for mo in months]

    def col_for(ym: str) -> int:
        if ym in month_keys:
            return month_keys.index(ym)
        return 0 if ym < month_keys[0] else len(month_keys) - 1

    n_cols = len(month_keys)
    phases_out: list[dict] = []
    for p in timeline_phases:
        pc = dict(p)
        s, e = p.get("start_date"), p.get("end_date")
        if s and e:
            pc["start_col"] = col_for(s[:7])
            pc["end_col"] = col_for(e[:7])
        else:
            pc["start_col"] = 0
            pc["end_col"] = n_cols - 1
        milestones_in = p.get("milestones") or []
        milestones_out: list[dict] = []
        for m in milestones_in:
            mc = dict(m)
            start_ym = m.get("start_ym")
            end_ym = m.get("end_ym")
            if start_ym is not None and end_ym is not None:
                mc["start_col"] = max(0, min(col_for(start_ym), n_cols - 1))
                mc["end_col"] = max(0, min(col_for(end_ym), n_cols - 1))
                if mc["end_col"] < mc["start_col"]:
                    mc["end_col"] = mc["start_col"]
            else:
                mc["start_col"] = pc["start_col"]
                mc["end_col"] = pc["end_col"]
            milestones_out.append(mc)
        pc["milestones"] = milestones_out
        phases_out.append(pc)
    return {"months": months, "phases": phases_out}


def _timeline_now_month_index(month_keys: list[str], today: date | None = None) -> int:
    """Return 0-based column index for *today* in the timeline months; -1 if before/after or empty."""
    if not month_keys:
        return -1
    d = today or date.today()
    ym = d.strftime("%Y-%m")
    if ym in month_keys:
        return month_keys.index(ym)
    if ym < month_keys[0]:
        return 0
    return len(month_keys) - 1


def _session_deadlines_for_issue() -> list[dict]:
    """Build list of deadline dicts (date, chamber, description) for The Issue FAQ, sorted by date."""
    deadlines = get_all_deadlines()
    out = [
        {"date": ev["date"], "chamber": chamber, "description": ev["description"]}
        for chamber, ev in deadlines
    ]
    out.sort(key=lambda d: (d["date"], d["chamber"]))
    return out


def _faq_session_and_deadlines_with_tooltips(
    session_deadlines: list[dict],
) -> tuple[dict, list[dict]]:
    """Build FAQ session block and deadlines with inline glossary tooltips (SESSION_SCHEDULE_TERMS + KEI).
    Returns (faq_session, deadlines) where faq_session items have answer_html and deadlines have description_html."""
    terms = _inline_glossary_terms(include_domain=False)
    faq_session = {
        "title": FAQ_SESSION["title"],
        "intro": FAQ_SESSION.get("intro"),
        "items": [],
    }
    for item in FAQ_SESSION.get("items") or []:
        answer = item.get("answer") or ""
        blocks = [answer]
        result = apply_inline_glossary(blocks, terms)
        faq_session["items"].append({**item, "answer_html": result[0] if result else answer})
    deadlines_out = []
    for d in session_deadlines:
        desc = d.get("description") or ""
        result = apply_inline_glossary([desc], terms)
        deadlines_out.append({**d, "description_html": result[0] if result else desc})
    return faq_session, deadlines_out


def _the_issue_blocks_for_glossary(
    constituent_brief: dict,
) -> tuple[list[str], list[tuple[str, int, int]]]:
    """Build flat list of text blocks for The Issue (intro, points, section paragraphs/bullets) and mapping for result indices.
    Returns (blocks, mapping) where mapping is list of ('intro'|'point'|'para'|'bullet', section_ix, sub_ix)."""
    blocks: list[str] = []
    mapping: list[tuple[str, int, int]] = []
    sections = constituent_brief.get("sections") or []
    if not sections:
        return blocks, mapping
    first_paras = sections[0].get("paragraphs") or []
    if not first_paras:
        return blocks, mapping
    intro = first_paras[0]
    blocks.append(intro)
    mapping.append(("intro", 0, 0))
    points = STRATEGIC_FIVE_POINTS or []
    for i, pt in enumerate(points):
        blocks.append(pt)
        mapping.append(("point", -1, i))
    for sec_ix, sec in enumerate(sections):
        paras = sec.get("paragraphs") or []
        for p_ix, p in enumerate(paras):
            if sec_ix == 0 and p_ix == 0:
                continue
            blocks.append(p)
            mapping.append(("para", sec_ix, p_ix))
        for b_ix, b in enumerate(sec.get("bullets") or []):
            blocks.append(b)
            mapping.append(("bullet", sec_ix, b_ix))
    return blocks, mapping


def _apply_the_issue_glossary(constituent_brief: dict) -> None:
    """Mutate constituent_brief: add intro_html, strategic_five_points_html, and per-section paragraphs_html, bullets_html."""
    blocks, mapping = _the_issue_blocks_for_glossary(constituent_brief)
    if not blocks:
        return
    terms = _inline_glossary_terms(include_domain=False)
    result = apply_inline_glossary(blocks, terms)
    sections = constituent_brief.get("sections") or []
    for sec in sections:
        sec["paragraphs_html"] = list(sec.get("paragraphs") or [])
        sec["bullets_html"] = list(sec.get("bullets") or [])
    intro_html = None
    points_html: list[str] = []
    for i, (kind, sec_ix, sub_ix) in enumerate(mapping):
        if i >= len(result):
            break
        if kind == "intro":
            intro_html = result[i]
        elif kind == "point":
            points_html.append(result[i])
        elif (
            kind == "para"
            and sec_ix < len(sections)
            and sub_ix < len(sections[sec_ix]["paragraphs_html"])
        ):
            sections[sec_ix]["paragraphs_html"][sub_ix] = result[i]
        elif (
            kind == "bullet"
            and sec_ix < len(sections)
            and sub_ix < len(sections[sec_ix]["bullets_html"])
        ):
            sections[sec_ix]["bullets_html"][sub_ix] = result[i]
    if intro_html is not None:
        constituent_brief["intro_html"] = intro_html
        if sections and sections[0]["paragraphs_html"]:
            sections[0]["paragraphs_html"][0] = intro_html
    if points_html:
        constituent_brief["strategic_five_points_html"] = points_html


@router.get("/the-issue", include_in_schema=False)
async def the_issue_page(
    request: Request,
    user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """Serve The Issue page: kei vehicle registration problem and how to help. Content from canonical .txt when present."""
    from ..kei_poll_context import get_kei_poll_sidebar_context

    brief_state_map_status = {
        s["state_abbr"]: _brief_map_fill_status(s) for s in BRIEF_STATE_STATUS
    }
    aamva_fix_abbrs = _brief_aamva_fix_state_abbrs()
    constituent_brief = _load_constituent_brief()
    if constituent_brief:
        _apply_the_issue_glossary(constituent_brief)
    try:
        raw_deadlines = _session_deadlines_for_issue()
    except (FileNotFoundError, ValueError):
        raw_deadlines = []
    faq_session, session_deadlines = _faq_session_and_deadlines_with_tooltips(raw_deadlines)
    ctx = {
        "request": request,
        "constituent_brief": constituent_brief,
        "fact_sheet_pdf_url": FACT_SHEET_PDF_URL,
        "faq_law": FAQ_LAW,
        "faq_advocacy": FAQ_ADVOCACY,
        "faq_session": faq_session,
        "session_deadlines": session_deadlines,
        "session_schedule_terms": SESSION_SCHEDULE_TERMS,
        "brief_state_status": BRIEF_STATE_STATUS,
        "brief_state_map_status_json": json.dumps(brief_state_map_status),
        "brief_aamva_fix_state_abbrs_json": json.dumps(aamva_fix_abbrs),
        "issue_sources": ISSUE_SOURCES,
        "strategic_mission": get_campaign_config().strategic_mission or STRATEGIC_MISSION,
        "strategic_vision": STRATEGIC_VISION,
        "strategic_five_points": STRATEGIC_FIVE_POINTS,
    }
    ctx.update(await get_kei_poll_sidebar_context(request, user, db))
    ctx["marquee_items"] = await get_marquee_items(db)
    return templates.TemplateResponse("the_issue.html", ctx)


def _apply_legislator_brief_glossary(legislator_brief: dict) -> None:
    """Mutate legislator_brief: add issue_one_sentence_html, core_ambiguity_html, and per-section paragraphs_html."""
    blocks: list[str] = []
    blocks.append(legislator_brief.get("issue_one_sentence", ""))
    blocks.append(legislator_brief.get("core_ambiguity", ""))
    for sec in legislator_brief.get("sections") or []:
        blocks.extend(sec.get("paragraphs") or [])
    if not blocks:
        return
    terms = _inline_glossary_terms(include_domain=False)
    result = apply_inline_glossary(blocks, terms)
    idx = 0
    if idx < len(result):
        legislator_brief["issue_one_sentence_html"] = result[idx]
        idx += 1
    if idx < len(result):
        legislator_brief["core_ambiguity_html"] = result[idx]
        idx += 1
    for sec in legislator_brief.get("sections") or []:
        paras = sec.get("paragraphs") or []
        sec["paragraphs_html"] = []
        for _ in paras:
            if idx < len(result):
                sec["paragraphs_html"].append(result[idx])
                idx += 1
            else:
                break


@router.get("/legislator-brief", include_in_schema=False)
async def legislator_brief_page(
    request: Request,
    user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """Serve the Legislator Brief: concise briefing for legislators and staff. Content from canonical .txt when present."""
    from ..kei_poll_context import get_kei_poll_sidebar_context

    brief_state_map_status = {
        s["state_abbr"]: _brief_map_fill_status(s) for s in BRIEF_STATE_STATUS
    }
    aamva_fix_abbrs = _brief_aamva_fix_state_abbrs()
    legislator_brief = _load_legislator_brief()
    if legislator_brief:
        _apply_legislator_brief_glossary(legislator_brief)
    ctx = {
        "request": request,
        "legislator_brief": legislator_brief,
        "strategic_five_points": STRATEGIC_FIVE_POINTS,
        "brief_documents": BRIEF_DOCUMENTS,
        "brief_state_status": BRIEF_STATE_STATUS,
        "brief_state_map_status_json": json.dumps(brief_state_map_status),
        "brief_aamva_fix_state_abbrs_json": json.dumps(aamva_fix_abbrs),
        "brief_bills_passed": BRIEF_BILLS_PASSED,
        "brief_bills_current": BRIEF_BILLS_CURRENT,
        "brief_sources": BRIEF_SOURCES,
        "faq": FAQ_LEGISLATORS,
    }
    ctx.update(await get_kei_poll_sidebar_context(request, user, db))
    return templates.TemplateResponse("legislator_brief.html", ctx)


@router.get("/fact-sheet", include_in_schema=False)
async def fact_sheet_page(
    request: Request,
    user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """Serve the one-page fact sheet for volunteers (Hardball Ch7; content from focused-next-steps doc §5)."""
    from ..kei_poll_context import get_kei_poll_sidebar_context

    fact_sheet_faq_ids = ("adv1", "adv1b", "adv2", "adv3", "adv4", "adv7")
    fact_sheet_faq_items = [i for i in FAQ_ADVOCACY["items"] if i["id"] in fact_sheet_faq_ids]
    ctx = {
        "request": request,
        "strategic_five_points": STRATEGIC_FIVE_POINTS,
        "fact_sheet_issue": FACT_SHEET_ISSUE,
        "fact_sheet_position": FACT_SHEET_POSITION,
        "fact_sheet_faq_items": fact_sheet_faq_items,
    }
    ctx.update(await get_kei_poll_sidebar_context(request, user, db))
    return templates.TemplateResponse("fact_sheet.html", ctx)


@router.get("/coalition", include_in_schema=False)
async def coalition_page(request: Request):
    """Serve the Supporting legislators / coalition page: recognized offices that engage."""
    return templates.TemplateResponse(
        "coalition.html",
        {"request": request},
    )


def _timeline_phases_with_inline_glossary() -> list[dict]:
    """Return a copy of TIMELINE_PHASES with summary_html and milestone title_html/description_html from inline glossary."""
    blocks: list[str] = []
    mapping: list[
        tuple[str, int, int]
    ] = []  # ('label'|'summary'|'title'|'desc', phase_ix, milestone_ix or -1)
    for ph_ix, phase in enumerate(TIMELINE_PHASES):
        blocks.append(phase.get("label", ""))
        mapping.append(("label", ph_ix, -1))
        blocks.append(phase.get("summary", ""))
        mapping.append(("summary", ph_ix, -1))
        for m_ix, m in enumerate(phase.get("milestones") or []):
            blocks.append(m.get("title", ""))
            mapping.append(("title", ph_ix, m_ix))
            blocks.append(m.get("description", ""))
            mapping.append(("desc", ph_ix, m_ix))
    terms = _inline_glossary_terms(include_domain=True)
    result = apply_inline_glossary(blocks, terms)
    out: list[dict] = []
    for ph_ix, phase in enumerate(TIMELINE_PHASES):
        ph_copy = dict(phase)
        ph_copy["milestones"] = [dict(m) for m in phase.get("milestones") or []]
        out.append(ph_copy)
    for idx, (kind, ph_ix, m_ix) in enumerate(mapping):
        if idx >= len(result):
            break
        if ph_ix >= len(out):
            continue
        if kind == "label":
            out[ph_ix]["label_html"] = result[idx]
        elif kind == "summary":
            out[ph_ix]["summary_html"] = result[idx]
        elif kind == "title" and m_ix >= 0 and m_ix < len(out[ph_ix]["milestones"]):
            out[ph_ix]["milestones"][m_ix]["title_html"] = result[idx]
        elif kind == "desc" and m_ix >= 0 and m_ix < len(out[ph_ix]["milestones"]):
            out[ph_ix]["milestones"][m_ix]["description_html"] = result[idx]
    return out


@router.get("/timeline", include_in_schema=False)
async def timeline_page(request: Request):
    """Serve the 2027 campaign master timeline: Feb 2026 through bill signed (waterfall/Gantt)."""
    try:
        session_deadlines = _session_deadlines_for_issue()
    except (FileNotFoundError, ValueError):
        session_deadlines = []
    timeline_phases = _timeline_phases_with_inline_glossary()
    waterfall = _timeline_waterfall_data(timeline_phases)
    month_keys = [mo["key"] for mo in waterfall["months"]]
    now_month_index = _timeline_now_month_index(month_keys)
    return templates.TemplateResponse(
        "timeline.html",
        {
            "request": request,
            "timeline_months": waterfall["months"],
            "timeline_phases": waterfall["phases"],
            "current_phase_id": _current_timeline_phase_id(),
            "now_month_index": now_month_index,
            "session_deadlines": session_deadlines,
            "session_label": session_label(),
        },
    )


@router.get("/glossary", include_in_schema=False)
async def glossary_page(
    request: Request,
    user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """Serve the definitions/glossary page: domain terms and kei vehicle terms."""
    from ..kei_poll_context import get_kei_poll_sidebar_context

    ctx = {
        "request": request,
        "domain_glossary": DOMAIN_GLOSSARY,
        "kei_glossary": KEI_GLOSSARY,
        "session_schedule_terms": SESSION_SCHEDULE_TERMS,
    }
    ctx.update(await get_kei_poll_sidebar_context(request, user, db))
    return templates.TemplateResponse("glossary.html", ctx)


@router.get("/privacy", include_in_schema=False)
async def privacy_page(request: Request):
    """Serve the Privacy policy page."""
    return templates.TemplateResponse("privacy.html", {"request": request})


@router.get("/terms", include_in_schema=False)
async def terms_page(request: Request):
    """Serve the Terms of use page."""
    return templates.TemplateResponse("terms.html", {"request": request})
