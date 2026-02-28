"""Legislative Power Map: explore page and graph data API."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from .. import advocacy_helpers as ah
from .. import config as cfg
from ..app_state import state
from ..constants import CATEGORY_CHOICES, CATEGORY_COMMITTEES, KEI_STATUS_OPTIONS
from ..member_lookup import find_member_by_district
from ..routers.content import STRATEGIC_FIVE_POINTS
from ..session_schedule import get_milestone_by_id, get_next_deadline_safe

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
router = APIRouter()
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
templates.env.globals["dev_available"] = cfg.DEV_MODE
templates.env.globals["app_base_url"] = cfg.APP_BASE_URL
templates.env.globals["site_name"] = cfg.SITE_NAME
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

from ..campaign_helpers import get_current_action_campaign_for_template  # noqa: E402

templates.env.globals["get_current_action_campaign"] = get_current_action_campaign_for_template
templates.env.globals["get_milestone_by_id"] = get_milestone_by_id
templates.env.globals["get_next_deadline"] = get_next_deadline_safe
templates.env.globals["kei_status_options"] = KEI_STATUS_OPTIONS


@router.get("/explore")
async def explore_page(request: Request):
    """Render the interactive Legislative Power Map."""
    return templates.TemplateResponse(
        "explore.html",
        {
            "request": request,
            "title": "Legislative Power Map",
            "categories": CATEGORY_CHOICES,
        },
    )


@router.get("/api/graph")
async def graph_data(
    topic: str = "",
    zip: str = "",
    focus: str = "relevant",
):
    """Return graph data (nodes + edges) for the Legislative Power Map.

    Query params:
    - topic: policy category name (e.g. "Transportation") — highlights
      members on relevant committees.
    - zip: Illinois ZIP code — identifies the user's senator and
      representative.
    - focus: "relevant" (default) — only top influencers + topic + your
      legislators; "all" — all 180 members.

    Returns JSON with nodes, edges, your_legislators, topic_committees, meta.
    """
    topic = topic.strip()
    committee_codes = CATEGORY_COMMITTEES.get(topic, [])
    topic_member_ids: set[str] = set()
    topic_committees_list: list[dict] = []
    if committee_codes:
        for code in committee_codes:
            cmembers: list[str] = []
            for role in state.committee_rosters.get(code, []):
                if role.member_id:
                    topic_member_ids.add(role.member_id)
                    cmembers.append(role.member_id)
            cmt = state.committee_lookup.get(code)
            topic_committees_list.append(
                {
                    "code": code,
                    "name": cmt.name if cmt else code,
                    "member_ids": cmembers,
                }
            )

    zip_code = zip.strip()
    your_senator_id: str | None = None
    your_rep_id: str | None = None
    if zip_code:
        district_info = state.zip_to_district.get(zip_code)
        if district_info:
            if district_info.il_senate:
                sen = find_member_by_district(state, "senate", district_info.il_senate)
                if sen:
                    your_senator_id = sen.id
            if district_info.il_house:
                rep = find_member_by_district(state, "house", district_info.il_house)
                if rep:
                    your_rep_id = rep.id

    your_legislator_ids = set()
    if your_senator_id:
        your_legislator_ids.add(your_senator_id)
    if your_rep_id:
        your_legislator_ids.add(your_rep_id)

    nodes: list[dict] = []
    for member in state.members:
        mb = state.moneyball.profiles.get(member.id) if state.moneyball else None
        ip = state.influence.get(member.id)

        influence_score = ip.influence_score if ip else (mb.moneyball_score if mb else 0.0)
        influence_label = ip.influence_label if ip else ""

        member_committees: list[dict] = []
        for cr in state.member_committee_roles.get(member.id, []):
            member_committees.append(
                {
                    "code": cr.get("code", ""),
                    "name": cr.get("name", ""),
                    "role": cr.get("role", ""),
                    "is_leadership": cr.get("is_leadership", False),
                }
            )

        party_abbr = ah.party_abbr_for_member(member)

        is_topic_relevant = member.id in topic_member_ids if topic_member_ids else False
        is_your_legislator = member.id in your_legislator_ids

        nodes.append(
            {
                "id": member.id,
                "name": member.name,
                "party": party_abbr,
                "chamber": member.chamber,
                "district": member.district,
                "influence_score": round(influence_score, 2),
                "influence_label": influence_label,
                "moneyball_score": round(mb.moneyball_score, 2) if mb else 0.0,
                "moneyball_rank": mb.rank_chamber if mb else 0,
                "is_leadership": mb.is_leadership if mb else False,
                "role": member.role or "",
                "committees": member_committees,
                "laws_passed": mb.laws_passed if mb else 0,
                "laws_filed": mb.laws_filed if mb else 0,
                "bridge_score": round(mb.bridge_score, 4) if mb else 0.0,
                "effectiveness_rate": round(mb.effectiveness_rate, 4) if mb else 0.0,
                "is_topic_relevant": is_topic_relevant,
                "is_your_legislator": is_your_legislator,
                "influence_signals": ip.influence_signals if ip else [],
            }
        )

    RELEVANT_TOP_N = 50
    if focus.strip().lower() == "relevant":
        if topic_member_ids:
            relevant_ids = topic_member_ids | your_legislator_ids
        else:
            by_influence = sorted(nodes, key=lambda n: n["influence_score"], reverse=True)
            relevant_ids = your_legislator_ids | {n["id"] for n in by_influence[:RELEVANT_TOP_N]}
        nodes = [n for n in nodes if n["id"] in relevant_ids]

    edges: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()
    adjacency = state.cosponsor_adjacency

    node_influence: dict[str, float] = {}
    for n in nodes:
        node_influence[n["id"]] = n["influence_score"]

    important_ids = topic_member_ids | your_legislator_ids
    top_by_influence = sorted(nodes, key=lambda n: n["influence_score"], reverse=True)[:20]
    important_ids |= {n["id"] for n in top_by_influence}

    MAX_EDGES_PER_NODE = 8

    for member_id, peers in adjacency.items():
        is_important = member_id in important_ids

        if is_important:
            target_peers = peers
        else:
            sorted_peers = sorted(
                peers,
                key=lambda pid: node_influence.get(pid, 0),
                reverse=True,
            )
            target_peers = sorted_peers[:MAX_EDGES_PER_NODE]

        for peer_id in target_peers:
            edge_key = tuple(sorted((member_id, peer_id)))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append(
                    {
                        "source": member_id,
                        "target": peer_id,
                    }
                )

    node_ids = {n["id"] for n in nodes}
    edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]

    return {
        "nodes": nodes,
        "edges": edges,
        "your_legislators": {
            "senator": your_senator_id,
            "representative": your_rep_id,
        },
        "topic_committees": topic_committees_list,
        "meta": {
            "total_members": len(nodes),
            "total_edges": len(edges),
            "topic": topic,
            "zip": zip_code,
            "focus": focus.strip().lower() or "all",
        },
    }
