"""Advocacy page helpers: card building, script/email copy, Power Broker selection.

All functions that need app state take `state` as the first argument (the same
AppState instance used in main.py lifespan). Pure helpers (script/email text,
stats sentence) take no state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .metrics_definitions import MONEYBALL_ONE_LINER
from .models import Member
from .moneyball import compute_power_badges

# ── Caller profile (personalization) ─────────────────────────────────────────


@dataclass
class CallerProfile:
    """Kei status + impact + optional personal note for script/email personalization."""

    kei_status: str | None  # from users.kei_status or cookie
    kei_impact_slug: str | None  # how it affects them
    kei_personal_note: str | None  # optional free-text sentence (max 200 chars)


def build_personalized_opening(
    caller: CallerProfile | None,
    is_constituent: bool,
    title_label: str,
    legislator_last: str,
) -> str:
    """Opening line for call script; personalized by kei_status + impact, else generic.

    If caller has no kei_status/impact, returns the generic constituent/non-constituent opening.
    If kei_personal_note is set, appends it after the opening sentence.
    """
    if not caller or (not caller.kei_status and not caller.kei_impact_slug):
        return _generic_opening(is_constituent, title_label)

    base = _personalized_opening_sentence(caller.kei_status, caller.kei_impact_slug, is_constituent)
    if not base:
        base = _generic_opening(is_constituent, title_label)
    else:
        base = SCRIPT_OPENING_NAME_LINE + base
    if caller.kei_personal_note and (note := caller.kei_personal_note.strip()):
        base = base.rstrip(".?") + ". " + (note if note.endswith(".") else note + ".")
    return base


SCRIPT_OPENING_NAME_LINE = "My name is [CALLER_NAME]. "

# Quick-context explainer for call script (after staff says yes). Canonical wording from KEI_GLOSSARY; sets the stage before THE PROBLEM.
SCRIPT_KEI_EXPLAINER_QUESTION = "Ever heard of Kei vehicles?"
SCRIPT_KEI_EXPLAINER_SHORT = (
    "They're a Japanese vehicle class — compact trucks, vans, and cars built to strict size and engine limits. "
    "Built for the highway in Japan, and federally legal to import here once they're 25 years old."
)


def _generic_opening(is_constituent: bool, title_label: str) -> str:
    if is_constituent:
        return (
            SCRIPT_OPENING_NAME_LINE
            + f"Hi — I'm a constituent and I'd like to leave a quick message for the {title_label}. "
            "It's about an issue that's affecting people in the district."
        )
    return (
        SCRIPT_OPENING_NAME_LINE + "Hi — I'm calling about kei vehicle registration in Illinois. "
        f"I'd like to leave a quick message for the {title_label} — it's something a lot of folks are running into."
    )


def build_personalized_email_why(caller: CallerProfile | None) -> str:
    """Sentence for 'This matters to me because X' in email body. Returns [ONE_SENTENCE_WHY] if no profile."""
    if not caller or (not caller.kei_status and not caller.kei_impact_slug):
        return "[ONE_SENTENCE_WHY]"
    reason = _personalized_reason_sentence(caller.kei_status, caller.kei_impact_slug)
    if not reason:
        return "[ONE_SENTENCE_WHY]"
    if caller.kei_personal_note and (note := caller.kei_personal_note.strip()):
        reason = reason.rstrip(".?") + ". " + (note if note.endswith(".") else note + ".")
    return reason


def _personalized_reason_sentence(kei_status: str | None, kei_impact_slug: str | None) -> str:
    """Just the reason clause (no 'I'm a constituent and') for email or script."""
    if not kei_status:
        return ""
    if kei_status == "registered":
        if kei_impact_slug == "work_commute":
            return "I own a Kei vehicle I use for my daily commute."
        if kei_impact_slug == "recreation":
            return "I'm a Kei owner and use mine for recreational purposes."
        if kei_impact_slug == "worried_revoked":
            return "I have a registered Kei and I'm worried the same thing could happen to me."
        return "I own a Kei vehicle that's currently registered."
    if kei_status == "revoked":
        if kei_impact_slug == "sitting_unused":
            return "My Kei registration was revoked and it's sitting in my garage."
        if kei_impact_slug == "lost_commute":
            return "My Kei registration was revoked and I lost my way to get to work."
        if kei_impact_slug == "cost_money":
            return "My Kei registration was revoked and it's cost me real money."
        return "My Kei vehicle registration was revoked."
    if kei_status == "denied":
        return "I was denied Kei registration under a confusing reading of state law."
    if kei_status == "would_want":
        if kei_impact_slug == "for_work":
            return (
                "I would buy a Kei vehicle for work but the registration ambiguity is stopping me."
            )
        if kei_impact_slug == "recreation":
            return "I'd like to get a Kei for recreation but the legal uncertainty holds me back."
        if kei_impact_slug == "small_business":
            return "the Kei registration issue affects my small business."
        return "I don't have a Kei yet but would want one — the current law is blocking that."
    if kei_status == "would_not_want":
        if kei_impact_slug == "support_cause":
            return "I support fixing this for Illinois residents who are affected."
        if kei_impact_slug == "know_someone":
            return "I know someone whose Kei was denied or revoked."
        if kei_impact_slug == "civic_duty":
            return "I think this is a fairness issue that the legislature should fix."
        return "I think this is an important fix for the state."
    return ""


def _personalized_opening_sentence(
    kei_status: str | None, kei_impact_slug: str | None, is_constituent: bool
) -> str:
    if not kei_status:
        return ""
    lead = "Hi — I'm in your district and " if is_constituent else "Hi — I'm calling because "
    if kei_status == "registered":
        if kei_impact_slug == "work_commute":
            return (
                lead
                + "I drive a Kei for my daily commute. There's something going on that's affecting folks like me — I'd like to leave a quick message for the office."
            )
        if kei_impact_slug == "recreation":
            return (
                lead
                + "I own a Kei and use it for fun — camping, runs to the store, that kind of thing. I wanted to reach out about an issue that's hitting a lot of us."
            )
        if kei_impact_slug == "worried_revoked":
            return (
                lead
                + "I've got a Kei that's registered right now, but I'm worried the same thing could happen to me that's happened to others. I'd like to leave a quick message."
            )
        return (
            lead
            + "I actually own a Kei that's registered. There's an issue affecting people like me in the district — I'd like to leave a quick message."
        )
    if kei_status == "revoked":
        if kei_impact_slug == "sitting_unused":
            return (
                lead
                + "my Kei registration was revoked and now it's just sitting in my garage. I'd like to leave a message about what's going on."
            )
        if kei_impact_slug == "lost_commute":
            return (
                lead
                + "my Kei registration was revoked and I lost my way to get to work. I'd like to leave a message so the office knows how this is affecting people."
            )
        if kei_impact_slug == "cost_money":
            return (
                lead
                + "my Kei registration was revoked and it's cost me real money. I wanted to reach out and leave a quick message."
            )
        return (
            lead
            + "my Kei registration was revoked. I'd like to leave a message about what's happening so the office is aware."
        )
    if kei_status == "denied":
        return (
            lead
            + "I was denied Kei registration — the whole thing was really confusing. I'd like to leave a quick message for the office."
        )
    if kei_status == "would_want":
        if kei_impact_slug == "for_work":
            return (
                lead
                + "I'd love to get a Kei for work but the registration situation is holding me back. I'd like to leave a message about it."
            )
        if kei_impact_slug == "recreation":
            return (
                lead
                + "I'd like to get a Kei for recreation but the legal uncertainty has me stuck. I'd like to leave a quick message."
            )
        if kei_impact_slug == "small_business":
            return (
                lead
                + "this Kei registration issue is affecting my small business. I'd like to leave a message."
            )
        return (
            lead
            + "I don't have a Kei yet but I'd want one — the way the law reads right now is blocking that. I'd like to leave a quick message."
        )
    if kei_status == "would_not_want":
        if kei_impact_slug == "support_cause":
            return (
                lead
                + "I care about fixing this for the people who are affected. I'd like to leave a quick message for the office."
            )
        if kei_impact_slug == "know_someone":
            return (
                lead
                + "I know someone whose Kei was denied or revoked. I wanted to reach out and leave a message."
            )
        if kei_impact_slug == "civic_duty":
            return (
                lead
                + "I think this is a fairness issue the legislature should fix. I'd like to leave a quick message."
            )
        return (
            lead + "I think this is an important fix for the state and I'd like to leave a message."
        )
    return ""


# ── Constants ───────────────────────────────────────────────────────────────

RECOMMENDATION_CHIP_PRIORITY: list[str] = [
    "cosponsor_lift",
    "outperforms_caucus",
    "bipartisan_reach",
    "big_network",
    "chair",
    "high_rank",
    "high_passage",
    "productive",
    "cross_party",
    "persuadable",
]


def party_abbr_for_member(member: Member | None) -> str:
    """Single-letter party abbreviation (R, D, or first char) for display."""
    if not member:
        return ""
    party = (member.party or "").lower()
    if "republican" in party:
        return "R"
    if "democrat" in party:
        return "D"
    return (member.party or "")[:1]


def get_preferred_phone_for_member(member: Member | None) -> str | None:
    """Prefer district office phone; fallback to first office with phone (e.g. Springfield)."""
    if not member:
        return None
    district_phone: str | None = None
    any_phone: str | None = None
    for office in member.offices:
        if not office.phone:
            continue
        any_phone = office.phone
        if "district" in office.name.lower():
            district_phone = office.phone
    return district_phone or any_phone


# ── State-taking helpers ──────────────────────────────────────────────────────


def committee_member_ids(state: Any, committee_codes: list[str]) -> set[str]:
    """Return a set of member IDs that sit on any of the given committees."""
    ids: set[str] = set()
    for code in committee_codes:
        for role in state.committee_rosters.get(code, []):
            ids.add(role.member_id)
    return ids


def build_influence_dict(state: Any, member: Member) -> dict | None:
    """Build influence data for a card from the influence engine state."""
    ip = state.influence.get(member.id)
    if not ip:
        return None

    piv = state.pivotality.get(member.name)
    sp = state.sponsor_pull.get(member.id)

    d: dict = {
        "score": ip.influence_score,
        "label": ip.influence_label,
        "rank_overall": ip.rank_overall,
        "rank_chamber": ip.rank_chamber,
        "signals": ip.influence_signals,
        "moneyball_pct": round(ip.moneyball_normalized * 100, 1),
        "betweenness_pct": round(ip.betweenness_normalized * 100, 1),
        "pivotality_pct": round(ip.pivotality_normalized * 100, 1),
        "pull_pct": round(ip.pull_normalized * 100, 1),
    }

    if piv:
        d["close_votes"] = piv.close_votes_total
        d["pivotal_winning"] = piv.pivotal_winning
        d["swing_votes"] = piv.swing_votes

    if sp:
        d["sponsor_lift"] = round(sp.sponsor_lift, 3)
        d["cosponsor_lift"] = round(sp.cosponsor_lift, 3)

    return d


def recommendation_chip_order(
    *,
    influence_dict: dict | None,
    committee_roles: list,
    voting_record_dict: dict | None,
    rank_chamber: int | None,
    chamber_size: int,
    passage_rate_pct: float | None,
    laws_passed: int | None,
    laws_filed: int | None,
    bridge_pct: float | None,
    relevant_committee_codes: list[str] | None = None,
) -> list[str]:
    """Return which recommendation chips apply, in ML-priority order.

    When relevant_committee_codes is set (e.g. topic + general committees),
    the chair chip is only included if the member chairs one of those committees.
    """
    net = influence_dict
    if relevant_committee_codes:
        has_chair = any(
            cr.get("is_leadership") and cr.get("code") in relevant_committee_codes
            for cr in (committee_roles or [])
        )
    else:
        has_chair = any(cr.get("is_leadership") for cr in (committee_roles or []))
    rank_percentile = (
        round((1 - (rank_chamber - 1) / chamber_size) * 100)
        if rank_chamber and chamber_size > 0
        else None
    )
    high_rank = rank_percentile is not None and rank_percentile >= 70
    is_persuadable = bool(voting_record_dict and voting_record_dict.get("is_persuadable"))
    high_passage = passage_rate_pct is not None and passage_rate_pct >= 15
    productive = (
        laws_passed is not None
        and laws_passed >= 3
        and (passage_rate_pct is None or passage_rate_pct < 15)
    )
    cross_party = bridge_pct is not None and bridge_pct >= 15

    cosponsor_lift = (
        net is not None
        and net.get("cosponsor_passage_multiplier") is not None
        and net["cosponsor_passage_multiplier"] >= 1.2
    )
    outperforms_caucus = (
        net is not None
        and net.get("passage_rate_vs_caucus") is not None
        and net["passage_rate_vs_caucus"] >= 1.2
    )
    bipartisan_reach = bool(net and net.get("bipartisan_label"))
    big_network = (
        net is not None
        and net.get("unique_collaborators") is not None
        and net["unique_collaborators"] >= 40
    )

    active = []
    if cosponsor_lift:
        active.append("cosponsor_lift")
    if outperforms_caucus:
        active.append("outperforms_caucus")
    if bipartisan_reach:
        active.append("bipartisan_reach")
    if big_network:
        active.append("big_network")
    if has_chair:
        active.append("chair")
    if high_rank:
        active.append("high_rank")
    if high_passage:
        active.append("high_passage")
    if productive:
        active.append("productive")
    if cross_party:
        active.append("cross_party")
    if is_persuadable:
        active.append("persuadable")

    priority_index = {k: i for i, k in enumerate(RECOMMENDATION_CHIP_PRIORITY)}
    return sorted(active, key=lambda k: priority_index.get(k, 999))


def member_to_card(
    state: Any,
    member: Member,
    *,
    why: str = "",
    badges: list[str] | None = None,
    relevant_committee_codes: list[str] | None = None,
) -> dict:
    """Convert a Member to a template-friendly dict for card rendering.

    script_hint and script_sections are left empty; callers set them via
    build_script_hint_* and build_script_sections_*.
    """
    phone = get_preferred_phone_for_member(member)

    mb = None
    if state.moneyball:
        mb = state.moneyball.profiles.get(member.id)

    laws_filed = mb.laws_filed if mb else None
    laws_passed = mb.laws_passed if mb else None
    passage_rate_pct = round((mb.effectiveness_rate * 100), 1) if mb and mb.laws_filed else None
    bridge_pct = round((mb.bridge_score * 100), 1) if mb else None

    sc = state.scorecards.get(member.id)
    scorecard_dict = None
    if sc is not None and (sc.primary_bill_count > 0 or sc.law_heat_score > 0):
        scorecard_dict = {
            "laws_filed": sc.law_heat_score,
            "laws_passed": sc.law_passed_count,
            "law_pass_rate_pct": round(sc.law_success_rate * 100, 1),
            "magnet_score": round(sc.magnet_score, 1),
            "bridge_pct": round(sc.bridge_score * 100, 1),
            "resolutions_filed": sc.resolutions_count,
            "resolutions_passed": sc.resolutions_passed_count,
            "resolution_pass_rate_pct": round(sc.resolution_pass_rate * 100, 1),
            "total_bills": sc.primary_bill_count,
            "total_passed": sc.passed_count,
            "overall_pass_rate_pct": round(sc.success_rate * 100, 1),
            "vetoed_count": sc.vetoed_count,
            "stuck_count": sc.stuck_count,
            "in_progress_count": sc.in_progress_count,
        }

    moneyball_dict = None
    if mb:
        moneyball_dict = {
            "effectiveness_rate_pct": round(mb.effectiveness_rate * 100, 1),
            "pipeline_depth_avg": round(mb.pipeline_depth_avg, 2),
            "magnet_score": round(mb.magnet_score, 1),
            "bridge_pct": round(mb.bridge_score * 100, 1),
            "network_centrality": round(mb.network_centrality, 3),
            "institutional_weight": round(mb.institutional_weight, 2),
        }

    influence_dict = None
    if mb and mb.unique_collaborators > 0:
        total_collab = (
            mb.collaborator_republicans + mb.collaborator_democrats + mb.collaborator_other
        )
        minority_share = (
            min(mb.collaborator_republicans, mb.collaborator_democrats) / total_collab
            if total_collab > 0
            else 0.0
        )
        if minority_share >= 0.3:
            bipartisan_label = "high bipartisan reach"
        elif minority_share >= 0.15:
            bipartisan_label = "moderate bipartisan reach"
        else:
            bipartisan_label = ""

        influence_dict = {
            "unique_collaborators": mb.unique_collaborators,
            "collaborator_republicans": mb.collaborator_republicans,
            "collaborator_democrats": mb.collaborator_democrats,
            "collaborator_other": mb.collaborator_other,
            "bipartisan_label": bipartisan_label,
            "magnet_score": round(mb.magnet_score, 1),
            "magnet_vs_chamber": mb.magnet_vs_chamber,
            "cosponsor_passage_rate_pct": round(mb.cosponsor_passage_rate * 100, 1),
            "cosponsor_passage_multiplier": mb.cosponsor_passage_multiplier,
            "chamber_median_cosponsor_rate_pct": round(
                mb.chamber_median_cosponsor_rate * 100,
                1,
            ),
            "passage_rate_vs_caucus": mb.passage_rate_vs_caucus,
            "caucus_avg_passage_rate_pct": round(mb.caucus_avg_passage_rate * 100, 1),
        }

    committee_roles = state.member_committee_roles.get(member.id, [])

    voting_record_dict: dict | None = None
    vr = state.member_vote_records.get(member.name)
    if vr and vr.total_votes > 0:
        voting_record_dict = {
            "total_votes": vr.total_votes,
            "total_floor_votes": vr.total_floor_votes,
            "total_committee_votes": vr.total_committee_votes,
            "yes_count": vr.yes_count,
            "no_count": vr.no_count,
            "present_count": vr.present_count,
            "nv_count": vr.nv_count,
            "yes_rate_pct": vr.yes_rate_pct,
            "party_alignment_pct": vr.party_alignment_pct,
            "party_defection_count": vr.party_defection_count,
            "is_persuadable": vr.party_defection_count > 0,
            "records": [
                {
                    "bill_number": r.bill_number,
                    "bill_description": r.bill_description,
                    "date": r.date,
                    "vote": r.vote,
                    "bill_status": r.bill_status,
                    "vote_type": r.vote_type,
                    "bill_status_url": r.bill_status_url,
                }
                for r in vr.records
            ],
        }

    power_badges_list: list[dict] = []
    chamber_size = 0
    if mb:
        chamber_size = (
            (
                len(state.moneyball.rankings_house)
                if member.chamber == "House"
                else len(state.moneyball.rankings_senate)
            )
            if state.moneyball
            else 0
        )
        raw_badges = compute_power_badges(mb, committee_roles, chamber_size)
        power_badges_list = [
            {
                "label": pb.label,
                "icon": pb.icon,
                "explanation": pb.explanation,
                "css_class": pb.css_class,
            }
            for pb in raw_badges
        ]

    rank_chamber = mb.rank_chamber if mb else None
    rank_percentile = None
    if mb and chamber_size > 0:
        rank_percentile = round((1 - (mb.rank_chamber - 1) / chamber_size) * 100)

    party_abbr = party_abbr_for_member(member)

    active_count = 0
    for bid in member.sponsored_bill_ids or []:
        b = state.bill_lookup.get(bid)
        if b and b.last_action:
            action_lower = b.last_action.lower()
            if not any(
                kw in action_lower
                for kw in (
                    "public act",
                    "effective date",
                    "vetoed",
                    "tabled",
                    "postponed indefinitely",
                    "session sine die",
                )
            ):
                active_count += 1

    recommendation_chip_order_result = recommendation_chip_order(
        influence_dict=influence_dict,
        committee_roles=committee_roles,
        voting_record_dict=voting_record_dict,
        rank_chamber=rank_chamber,
        chamber_size=chamber_size,
        passage_rate_pct=passage_rate_pct,
        laws_passed=laws_passed,
        laws_filed=laws_filed,
        bridge_pct=bridge_pct,
        relevant_committee_codes=relevant_committee_codes,
    )

    return {
        "name": member.name,
        "id": member.id,
        "district": member.district,
        "party": member.party,
        "party_abbr": party_abbr,
        "chamber": member.chamber,
        "role": member.role,
        "phone": phone,
        "email": member.email,
        "laws_filed": laws_filed,
        "laws_passed": laws_passed,
        "passage_rate_pct": passage_rate_pct,
        "bridge_score": round(mb.bridge_score, 4) if mb else None,
        "bridge_pct": bridge_pct,
        "moneyball_score": round(mb.moneyball_score, 2) if mb else None,
        "moneyball_explanation": MONEYBALL_ONE_LINER,
        "moneyball": moneyball_dict,
        "influence_network": influence_dict,
        "member_url": member.member_url,
        "photo_url": getattr(member, "photo_url", "") or "",
        "bio_text": (member.bio_text or "").strip(),
        "why": why,
        "badges": badges or [],
        "power_badges": power_badges_list,
        "script_hint": "",
        "scorecard": scorecard_dict,
        "committee_roles": committee_roles,
        "voting_record": voting_record_dict,
        "rank_chamber": rank_chamber,
        "chamber_size": chamber_size,
        "rank_percentile": rank_percentile,
        "active_bills": active_count,
        "influence": build_influence_dict(state, member),
        "recommendation_chip_order": recommendation_chip_order_result,
    }


# ── Pure helpers (no state) ────────────────────────────────────────────────────


def stats_sentence(card: dict) -> str:
    """Build a short stats clause from the card's empirical fields."""
    parts: list[str] = []
    if card.get("laws_filed") and card.get("laws_passed") is not None:
        parts.append(
            f"they've passed {card['laws_passed']} of {card['laws_filed']} laws "
            f"({card['passage_rate_pct'] or 0}% passage rate)"
        )
    if card.get("bridge_pct") is not None and card["bridge_pct"] > 0:
        parts.append(f"{card['bridge_pct']}% of their bills have cross-party co-sponsors")
    if parts:
        return (
            parts[0][0].upper()
            + parts[0][1:]
            + (" and " + parts[1] if len(parts) > 1 else "")
            + "."
        )
    return ""


def script_wow_line(card: dict) -> str:
    """One compelling, legislator-relevant line from influence network for script copy."""
    net = card.get("influence_network")
    if not net:
        return ""
    parts = []
    if net.get("cosponsor_passage_multiplier") and net["cosponsor_passage_multiplier"] >= 1.2:
        parts.append(
            f"Bills they co-sponsor tend to pass at {net['cosponsor_passage_multiplier']}\u00d7 the chamber rate"
        )
    if net.get("passage_rate_vs_caucus") and net["passage_rate_vs_caucus"] >= 1.2:
        parts.append(
            f"Their bills outperform their caucus by {net['passage_rate_vs_caucus']}\u00d7"
        )
    if net.get("bipartisan_label"):
        parts.append(f"They have {net['bipartisan_label']}")
    if net.get("unique_collaborators") and net["unique_collaborators"] >= 40:
        parts.append(f"They work with {net['unique_collaborators']}+ colleagues across the chamber")
    if not parts:
        return ""
    return " " + parts[0] + "."


# Three conclusions we want every legislator to remember (elevator version; Hardball Ch7).
# District-first order; full 5 points live in content.STRATEGIC_FIVE_POINTS.
SCRIPT_ELEVATOR_THREE: tuple[str, ...] = (
    "It's a real issue in your district — people have had legal vehicles denied registration.",
    "The fix is simple: one line of Illinois law clarified, not a new vehicle class or a big exemption.",
    "The one-pager shows which states have already fixed their Kei bans — Illinois can do the same.",
)

# Short call script (elevator pitch): problem + ask + get email. Used when caller toggles to "Short pitch."
SCRIPT_SHORT_PITCH = (
    "People in your district have had Kei registration denied. "
    "I'm asking the office to be aware — I can email a one-pager with the statute and which states "
    "have already fixed their Kei bans. Could I get the best email to send that to?"
)
SCRIPT_SHORT_PITCH_BROKER = (
    "People across the state have had Kei registration denied. "
    "I'm asking the office to be aware — I can email a one-pager with the statute and which states "
    "have already fixed their Kei bans. Could I get the best email to send that to?"
)


# Show "One of N Illinois residents" only when N = unique people who've done ≥1 outreach is ≥ this.
SOCIAL_PROOF_MIN_PEOPLE = 100

# Show "others have contacted this office" only when this many+ call/email events for this member.
OFFICE_CONTACT_MIN = 2


def _office_contact_phrase(contact_count_this_office: int) -> str:
    """Optional line when this office has had multiple constituents reach out. Only when count >= OFFICE_CONTACT_MIN."""
    if contact_count_this_office >= OFFICE_CONTACT_MIN:
        return " A few other constituents have already been in touch with your office about this."
    return ""


def _volume_phrase(calls_total: int) -> str:
    """Constituent volume signal for script and email (Hardball Ch7: tally sheet).
    Numeric social proof only when calls_total >= SOCIAL_PROOF_MIN_PEOPLE; else generic line.
    """
    if calls_total >= SOCIAL_PROOF_MIN_PEOPLE:
        return f"Over {calls_total} people across Illinois have already been in touch with legislators about this."
    return "Lots of Illinois residents have been reaching out to legislators about this."


def build_script_sections_senator(
    card: dict,
    zip_code: str,
    district: str,
    *,
    calls_total: int = 0,
    contact_count_this_office: int = 0,
) -> dict:
    wow = script_wow_line(card)
    volume = _volume_phrase(calls_total)
    office_line = _office_contact_phrase(contact_count_this_office)
    return {
        "opening": (
            SCRIPT_OPENING_NAME_LINE
            + "Hi — I'm a constituent and I'd like to leave a quick message for the Senator. "
            "It's about an issue that's affecting people in the district."
        ),
        "why_them": (
            "This is your state senator — your direct rep in Springfield. "
            "When constituents call, it gets noticed; yours counts." + wow
        ),
        "kei_explainer_question": SCRIPT_KEI_EXPLAINER_QUESTION,
        "kei_explainer_short": SCRIPT_KEI_EXPLAINER_SHORT,
        "the_problem": (
            "Kei vehicles are those compact trucks, vans, and cars from Japan — built for the highway, "
            "federally legal to import after 25 years. Here in Illinois though, people in your district have had "
            'registration denied or their titles stamped "Not Eligible for Registration." '
            + volume
            + office_line
        ),
        "the_problem_after_explainer": (
            "Here in Illinois, people in your district have had registration denied or their titles stamped "
            '"Not Eligible for Registration." ' + volume + office_line
        ),
        "the_legal_why": (
            "The state's reading 625 ILCS 5/3-401(c-1) in a way that excludes these vehicles. "
            "There's an ambiguity in the statute, not a clear ban — the General Assembly could clarify it so "
            "highway-built, federally legal imports can be registered here."
        ),
        "the_ask": (
            "I'm just asking the Senator to be aware of it. I can email a one-pager — two pages, "
            "has the statute and shows which states have already fixed their Kei bans. No commitment — "
            "I just want the office to have the facts."
        ),
        "easy_yes_close": ("Could I get your name and the best email to send that one-pager to?"),
        "closing": "Thanks so much for your time.",
        "conclusions": SCRIPT_ELEVATOR_THREE,
        "short_pitch": SCRIPT_SHORT_PITCH,
    }


def build_script_sections_rep(
    card: dict,
    zip_code: str,
    district: str,
    *,
    calls_total: int = 0,
    contact_count_this_office: int = 0,
) -> dict:
    wow = script_wow_line(card)
    volume = _volume_phrase(calls_total)
    office_line = _office_contact_phrase(contact_count_this_office)
    return {
        "opening": (
            SCRIPT_OPENING_NAME_LINE
            + "Hi — I'm a constituent and I'd like to leave a quick message for the Representative. "
            "It's about an issue that's affecting people in the district."
        ),
        "why_them": (
            "This is your state rep. They vote in the House before anything gets to the Senate, "
            "so having them aware of this really matters." + wow
        ),
        "kei_explainer_question": SCRIPT_KEI_EXPLAINER_QUESTION,
        "kei_explainer_short": SCRIPT_KEI_EXPLAINER_SHORT,
        "the_problem": (
            "Kei vehicles are those compact trucks, vans, and cars from Japan — built for the highway, "
            "federally legal to import after 25 years. Here in Illinois though, people in your district have had "
            'registration denied or their titles stamped "Not Eligible for Registration." '
            + volume
            + office_line
        ),
        "the_problem_after_explainer": (
            "Here in Illinois, people in your district have had registration denied or their titles stamped "
            '"Not Eligible for Registration." ' + volume + office_line
        ),
        "the_legal_why": (
            "The state's reading 625 ILCS 5/3-401(c-1) in a way that excludes these vehicles. "
            "There's an ambiguity in the statute, not a clear ban — the General Assembly could clarify it so "
            "highway-built, federally legal imports can be registered here."
        ),
        "the_ask": (
            "I'm just asking them to be aware of it. I can email a one-pager — two pages, "
            "has the statute and shows which states have already fixed their Kei bans. No commitment — "
            "I just want the office to have the facts."
        ),
        "easy_yes_close": ("Could I get your name and the best email to send that one-pager to?"),
        "closing": "Thanks so much for your time.",
        "conclusions": SCRIPT_ELEVATOR_THREE,
        "short_pitch": SCRIPT_SHORT_PITCH,
    }


def build_script_sections_broker(
    card: dict,
    broker_why: str,
    *,
    calls_total: int = 0,
    contact_count_this_office: int = 0,
) -> dict:
    is_chair = "Chair of the" in broker_why or "chairs the" in broker_why.lower()
    wow = script_wow_line(card)
    volume = _volume_phrase(calls_total)
    office_line = (
        " A few other constituents have already been in touch with this office about this."
        if contact_count_this_office >= OFFICE_CONTACT_MIN
        else ""
    )
    if is_chair:
        why = (
            "This legislator chairs the committee that decides whether the bill gets a hearing — "
            "they're the gatekeeper. Getting them on board can open the door." + wow
        )
    else:
        why = (
            "This senator has one of the highest influence scores in the chamber. "
            "When they co-sponsor, other members take notice." + wow
        )
    return {
        "opening": (
            SCRIPT_OPENING_NAME_LINE
            + "Hi — I'm calling about kei vehicle registration in Illinois. "
            "I'd like to leave a quick message for the Senator — it's something a lot of folks are running into."
        ),
        "why_them": why,
        "kei_explainer_question": SCRIPT_KEI_EXPLAINER_QUESTION,
        "kei_explainer_short": SCRIPT_KEI_EXPLAINER_SHORT,
        "the_problem": (
            "Kei vehicles are those compact trucks, vans, and cars from Japan — built for the highway, "
            "federally legal to import after 25 years. Across the state, people have had registration denied "
            'or their titles stamped "Not Eligible for Registration." ' + volume + office_line
        ),
        "the_problem_after_explainer": (
            "Across the state, people have had registration denied or their titles stamped "
            '"Not Eligible for Registration." ' + volume + office_line
        ),
        "the_legal_why": (
            "The state's reading 625 ILCS 5/3-401(c-1) in a way that excludes these vehicles. "
            "A narrow statutory clarification would fix it — and the one-pager shows which states "
            "have already overturned their Kei bans."
        ),
        "the_ask": (
            "I'm just asking them to be aware of it. I can email a one-pager — two pages, "
            "statute cite, and which states have fixed their Kei bans. No commitment — "
            "I just want the office to have the facts."
        ),
        "easy_yes_close": ("Could I get your name and the best email to send that one-pager to?"),
        "closing": "Thanks for your time.",
        "conclusions": SCRIPT_ELEVATOR_THREE,
        "short_pitch": SCRIPT_SHORT_PITCH_BROKER,
    }


def build_script_hint_senator(
    card: dict, zip_code: str, district: str, *, calls_total: int = 0
) -> str:
    s = build_script_sections_senator(card, zip_code, district, calls_total=calls_total)
    return " ".join(
        filter(
            None,
            [
                s["opening"],
                s["the_problem"],
                s["the_legal_why"],
                s["the_ask"],
                s["easy_yes_close"],
                s["closing"],
            ],
        )
    )


def build_script_hint_rep(card: dict, zip_code: str, district: str, *, calls_total: int = 0) -> str:
    s = build_script_sections_rep(card, zip_code, district, calls_total=calls_total)
    return " ".join(
        filter(
            None,
            [
                s["opening"],
                s["the_problem"],
                s["the_legal_why"],
                s["the_ask"],
                s["easy_yes_close"],
                s["closing"],
            ],
        )
    )


def build_script_hint_broker(card: dict, broker_why: str, *, calls_total: int = 0) -> str:
    s = build_script_sections_broker(card, broker_why, calls_total=calls_total)
    return " ".join(
        filter(
            None,
            [
                s["opening"],
                s["the_problem"],
                s["the_legal_why"],
                s["the_ask"],
                s["easy_yes_close"],
                s["closing"],
            ],
        )
    )


def build_email_subject(zip_code: str, district: str | None = None) -> str:
    """Legacy: card prefill subject (constituent variant)."""
    return build_email_subject_line(zip_code, variant="constituent", district=district)


def build_email_subject_line(
    zip_code: str, variant: str = "constituent", district: str | None = None
) -> str:
    """Subject line: constituent (optional district) or general. District in subject for staff triage."""
    if variant == "general":
        return "Request: Kei vehicle registration in Illinois — please review our one-pager"
    if district:
        return f"Constituent request (District {district}): Kei vehicle registration in Illinois — please review our one-pager"
    return "Constituent request: Kei vehicle registration in Illinois — please review our one-pager"


def _kei_email_body_core(
    title_label: str,
    legislator_last: str,
    *,
    greeting_line: str | None = None,
    one_pager_points: list[str] | None = None,
    calls_total: int = 0,
    district_label: str = "[DISTRICT]",
    caller: CallerProfile | None = None,
) -> str:
    """Kei vehicle email body (awareness stage): mad-lib placeholders, stage-correct ask.

    [LEGISLATOR_NAME], [CITY_OR_ZIP], [ONE_SENTENCE_WHY], [CALLER_NAME], [CALLER_PHONE], [CALLER_EMAIL].
    greeting_line: after-call use "Good [TIME_OF_DAY] [CONTACT_NAME],".
    calls_total: reserved for future social proof when unique outreach count >= SOCIAL_PROOF_MIN_PEOPLE.
    district_label: e.g. "Senate District 18" for constituent line.
    caller: when set, replaces [ONE_SENTENCE_WHY] with personalized reason.
    """
    first_line = greeting_line if greeting_line is not None else "Hi [LEGISLATOR_NAME],\n\n"
    one_sentence_why = build_personalized_email_why(caller)
    if one_pager_points:
        outlines = "".join(f"\t\u2022 {p}\n" for p in one_pager_points)
    else:
        outlines = (
            "\t\u2022 The specific statutory ambiguity being relied upon\n"
            "\t\u2022 Why this is a legislative clarification issue rather than an administrative one\n"
            "\t\u2022 A narrow, targeted fix that preserves all existing Illinois safety, equipment, "
            "and insurance requirements\n"
            "\t\u2022 Examples of other states that have authorized or restored registration through "
            "statute or formal policy\n"
        )
    volume_line = "Illinois residents have been reaching out to legislators about this."
    return (
        first_line + "I live in [CITY_OR_ZIP] and am a constituent of " + district_label + ".\n\n"
        "This matters to me because " + one_sentence_why + "\n\n"
        "I'm writing because real residents in your district have had lawfully imported kei vehicles \u2014 "
        "compact trucks, vans, and cars originally built for highway use in Japan, federally legal to import "
        "under the 25-year rule \u2014 denied registration at the Illinois SOS level or had their titles "
        'branded "Not Eligible for Registration." '
        "This is based on how Illinois interprets 625 ILCS 5/3-401(c-1), not on federal law or "
        "missing paperwork.\n\n"
        f"{volume_line} I wanted to make sure your office is aware.\n\n"
        "Attached is a two-page one-pager that outlines:\n"
        f"{outlines}\n"
        "The goal is not a new vehicle class or exemption \u2014 it's a narrow clarifying amendment to "
        "one line of existing law. The one-pager shows which states have successfully overturned their Kei bans.\n\n"
        "For this stage, I'm only asking your office to:\n"
        "\t\u2022 Review the one-pager \u2014 it takes about 3 minutes\n"
        "\t\u2022 If your office has questions or interest, I'm happy to connect further\n\n"
        "I know your office receives many requests. This one is straightforward \u2014 "
        "no new regulatory framework, no cost to the state \u2014 and it addresses an unfair "
        "situation for Illinois residents who did everything right.\n\n"
        "Thank you for your time.\n\n"
        "[CALLER_NAME]\n"
        "[CALLER_PHONE]\n"
        "[CALLER_EMAIL]\n"
    )


def build_email_body(
    member_name: str,
    script_hint: str,
    has_public_email: bool,
    *,
    chamber: str | None = None,
    district: str | None = None,
    one_pager_points: list[str] | None = None,
    calls_total: int = 0,
    caller: CallerProfile | None = None,
) -> str:
    """Legacy: card prefill body. Same Kei email as drawer."""
    if chamber:
        title_label, legislator_last, _, _, district_label = _legislator_email_context(
            member_name, chamber, district
        )
    else:
        title_label = "Rep./Sen."
        legislator_last = member_name.split()[-1] if member_name else "[Last Name]"
        district_label = "[DISTRICT]"
    return _kei_email_body_core(
        title_label,
        legislator_last,
        one_pager_points=one_pager_points,
        calls_total=calls_total,
        district_label=district_label,
        caller=caller,
    )


def _legislator_email_context(
    legislator_name: str,
    chamber: str | None,
    district: str | None,
) -> tuple[str, str, str, str, str]:
    """Return (title_label, legislator_last, legislator_full, office_name, district_label)."""
    title_label = "Senator" if chamber and chamber.lower() == "senate" else "Representative"
    short_title = "Sen." if chamber and chamber.lower() == "senate" else "Rep."
    legislator_full = legislator_name or "[LEGISLATOR_FULL]"
    legislator_last = (
        legislator_name.split()[-1] if legislator_name else ""
    ) or "[LEGISLATOR_LAST]"
    office_name = (
        f"Office of {short_title} {legislator_last}"
        if legislator_last != "[LEGISLATOR_LAST]"
        else "[OFFICE_NAME]"
    )
    if chamber and (chamber.lower() == "senate") and district:
        district_label = f"Senate District {district}"
    elif district:
        district_label = f"House District {district}"
    else:
        district_label = "[DISTRICT]"
    return title_label, legislator_last, legislator_full, office_name, district_label


def get_legislator_display_name(
    legislator_name: str,
    chamber: str | None = None,
    district: str | None = None,
) -> str:
    """Return display name for email greeting, e.g. 'Representative Smith' or 'Senator Jones'."""
    title_label, legislator_last, _, _, _ = _legislator_email_context(
        legislator_name, chamber, district
    )
    return f"{title_label} {legislator_last}".strip()


def legislator_drawer_context(member: Member | None) -> dict[str, str]:
    """Return template context for call/email drawer: title_label, legislator_last, office_name, district_label.
    Single source for Senator/Rep, district label, office name."""
    if not member:
        return {
            "title_label": "Representative",
            "legislator_last": "",
            "office_name": "[OFFICE_NAME]",
            "district_label": "[DISTRICT]",
        }
    title_label, legislator_last, _leg_full, office_name, district_label = (
        _legislator_email_context(
            member.name, getattr(member, "chamber", None), getattr(member, "district", None)
        )
    )
    return {
        "title_label": title_label,
        "legislator_last": legislator_last,
        "office_name": office_name,
        "district_label": district_label,
    }


def build_after_call_email_subject(zip_code: str, district: str | None = None) -> str:
    return build_email_subject_line(zip_code, variant="constituent", district=district)


def build_after_call_email_body(
    staffer_name: str,
    legislator_name: str,
    zip_code: str,
    *,
    chamber: str | None = None,
    district: str | None = None,
    target_type: str = "NON_COMMITTEE",
    call_date: str = "",
    one_pager_points: list[str] | None = None,
    calls_total: int = 0,
    caller: CallerProfile | None = None,
) -> str:
    """After-call follow-up email body (awareness stage, mad-lib placeholders).

    When staffer_name is provided, use "Good [TIME_OF_DAY] [CONTACT_NAME]," for greeting.
    """
    title_label, legislator_last, _, _, district_label = _legislator_email_context(
        legislator_name, chamber, district
    )
    greeting = None
    if (staffer_name or "").strip():
        greeting = "Good [TIME_OF_DAY] [CONTACT_NAME],\n\n"
    return _kei_email_body_core(
        title_label,
        legislator_last,
        greeting_line=greeting,
        one_pager_points=one_pager_points,
        calls_total=calls_total,
        district_label=district_label,
        caller=caller,
    )


def build_email_first_subject(zip_code: str, district: str | None = None) -> str:
    return build_email_subject_line(zip_code, variant="constituent", district=district)


def build_email_first_body(
    legislator_name: str,
    zip_code: str,
    *,
    chamber: str | None = None,
    district: str | None = None,
    target_type: str = "NON_COMMITTEE",
    one_pager_points: list[str] | None = None,
    calls_total: int = 0,
    caller: CallerProfile | None = None,
) -> str:
    """Email-first (no prior call) body (awareness stage, mad-lib placeholders)."""
    title_label, legislator_last, _, _, district_label = _legislator_email_context(
        legislator_name, chamber, district
    )
    return _kei_email_body_core(
        title_label,
        legislator_last,
        one_pager_points=one_pager_points,
        calls_total=calls_total,
        district_label=district_label,
        caller=caller,
    )


def find_power_broker(
    state: Any,
    *,
    exclude_senate_district: str = "",
    exclude_house_district: str = "",
    committee_ids: set[str] | None = None,
    committee_codes: list[str] | None = None,
    category_name: str = "",
) -> tuple[Member | None, str]:
    """Find the Power Broker: committee chair for the topic (default Transportation), or highest Moneyball senator or rep outside the user's district. Returns (Member | None, why_text)."""
    member_lookup = {m.id: m for m in state.members}

    if committee_codes:
        for code in committee_codes:
            for cmr in state.committee_rosters.get(code, []):
                role_lower = cmr.role.lower()
                if "chair" in role_lower and "vice" not in role_lower:
                    chair_member = member_lookup.get(cmr.member_id)
                    if chair_member and chair_member.district != (
                        exclude_senate_district
                        if chair_member.chamber == "Senate"
                        else exclude_house_district
                    ):
                        committee_name = ""
                        cmt = state.committee_lookup.get(code)
                        if cmt:
                            committee_name = cmt.name
                        parts = [f"Chair of the {committee_name or code} committee"]
                        if category_name:
                            parts.append(
                                f"the institutional gatekeeper for {category_name} legislation"
                            )
                        mb = (
                            state.moneyball.profiles.get(cmr.member_id) if state.moneyball else None
                        )
                        if mb:
                            parts.append(
                                f"Moneyball score: {mb.moneyball_score}, "
                                f"effectiveness: {mb.effectiveness_rate:.0%}"
                            )
                        why = ". ".join(parts) + "."
                        return chair_member, why

    if not state.moneyball:
        return None, ""

    best_profile = None
    for profile in state.moneyball.profiles.values():
        if profile.chamber == "Senate" and profile.district == exclude_senate_district:
            continue
        if profile.chamber == "House" and profile.district == exclude_house_district:
            continue
        if committee_ids and profile.member_id not in committee_ids:
            continue
        if best_profile is None or profile.moneyball_score > best_profile.moneyball_score:
            best_profile = profile

    if best_profile is None:
        return None, ""

    member = member_lookup.get(best_profile.member_id)
    if member is None:
        return None, ""

    chamber_label = "senator" if best_profile.chamber == "Senate" else "representative"
    parts = [
        f"Highest Moneyball score ({best_profile.moneyball_score}) "
        f"among senators and representatives outside your district ({chamber_label})",
    ]
    if category_name:
        parts.append(f"sits on a {category_name} committee")
    parts.append(
        f"effectiveness: {best_profile.effectiveness_rate:.0%}, "
        f"{best_profile.unique_collaborators} collaborators"
    )
    why = ". ".join(parts) + "."
    return member, why


def test_member_list(state: Any, max_count: int = 20) -> list[dict[str, Any]]:
    """First N members for test-mode dropdown/jump links."""
    out: list[dict[str, Any]] = []
    for m in state.members[:max_count]:
        out.append({"id": m.id, "name": m.name})
    return out
