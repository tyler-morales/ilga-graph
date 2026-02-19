"""Advocacy page helpers: card building, script/email copy, Power Broker and Ally selection.

All functions that need app state take `state` as the first argument (the same
AppState instance used in main.py lifespan). Pure helpers (script/email text,
stats sentence) take no state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .metrics_definitions import MONEYBALL_ONE_LINER
from .models import Member
from .moneyball import compute_power_badges

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
) -> list[str]:
    """Return which recommendation chips apply, in ML-priority order."""
    net = influence_dict
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
) -> dict:
    """Convert a Member to a template-friendly dict for card rendering.

    script_hint and script_sections are left empty; callers set them via
    build_script_hint_* and build_script_sections_*.
    """
    phone = None
    for office in member.offices:
        if office.phone:
            phone = office.phone
            break

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

    if "republican" in (member.party or "").lower():
        party_abbr = "R"
    elif "democrat" in (member.party or "").lower():
        party_abbr = "D"
    elif member.party:
        party_abbr = member.party[:1]
    else:
        party_abbr = ""

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
    """One compelling, positive line from influence network for script copy."""
    net = card.get("influence_network")
    if not net:
        return ""
    parts = []
    if net.get("cosponsor_passage_multiplier") and net["cosponsor_passage_multiplier"] >= 1.2:
        parts.append(
            f"Bills they co-sponsor pass at {net['cosponsor_passage_multiplier']}\u00d7 the chamber rate"
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


def build_script_sections_senator(card: dict, zip_code: str, district: str) -> dict:
    wow = script_wow_line(card)
    return {
        "opening": (
            f"Hi, I\u2019m a constituent calling from ZIP {zip_code}, Senate District {district}. "
            "I\u2019d like to leave a message for the Senator about kei truck legislation."
        ),
        "why_them": (
            "This is your state senator \u2014 your direct representative in Springfield. "
            "Constituent calls are tracked; yours counts." + wow
        ),
        "the_ask": (
            "I'm calling to ask for the Senator's support for kei truck legalization in Illinois. "
            "Please support [bill number / the bill] when it comes to a vote. Thank you."
        ),
        "closing": "Thank you for your time and your service.",
    }


def build_script_sections_rep(card: dict, zip_code: str, district: str) -> dict:
    wow = script_wow_line(card)
    return {
        "opening": (
            f"Hi, I\u2019m a constituent from ZIP {zip_code}, House District {district}. "
            "I\u2019d like to leave a message for the Representative about kei truck legislation."
        ),
        "why_them": (
            "This is your state representative. They vote on bills in the House before they reach "
            "the Senate, so their support is critical." + wow
        ),
        "the_ask": (
            "I'm calling to ask them to sponsor or support kei truck legislation. "
            "Please support [bill number / the bill] when it comes to a vote. Thank you."
        ),
        "closing": "Thank you for your time and your service.",
    }


def build_script_sections_broker(card: dict, broker_why: str) -> dict:
    is_chair = "Chair of the" in broker_why or "chairs the" in broker_why.lower()
    wow = script_wow_line(card)
    if is_chair:
        why = (
            "This legislator chairs the committee that controls whether the bill gets a hearing \u2014 "
            "they are the institutional gatekeeper. Getting their support can unlock the committee."
            + wow
        )
    else:
        why = (
            "This senator has one of the highest influence scores in the chamber. "
            "Their co-sponsorship signals to other members that the bill is serious." + wow
        )
    return {
        "opening": (
            "Hi, I\u2019m calling about kei truck legislation in Illinois. "
            "I\u2019d like to leave a message about co-sponsorship."
        ),
        "why_them": why,
        "the_ask": (
            "I'm asking them to co-sponsor the bill. Mention that constituents across the state "
            "support this, and that their support would help move it through the process."
        ),
        "closing": "Thank you for your time.",
    }


def build_script_sections_ally(card: dict) -> dict:
    bridge = card.get("bridge_pct")
    net = card.get("influence_network")
    bipartisan = (net or {}).get("bipartisan_label") or ""
    if bridge is not None and bridge >= 15:
        why = (
            f"This senator sits next to yours in the chamber and has a {bridge}% cross-party "
            "co-sponsorship rate \u2014 they regularly work across the aisle. "
            "A natural partner for your senator on this issue."
        )
    elif bipartisan:
        why = (
            "This senator sits physically next to yours in the chamber and has "
            f"{bipartisan}. They are a natural partner for your senator on this issue."
        )
    else:
        why = (
            "This senator sits physically next to yours in the chamber. "
            "Proximity matters \u2014 asking your senator to partner with them can build momentum."
        )
    return {
        "opening": (
            "Hi, I\u2019m calling about kei truck legislation. "
            "I\u2019d like to leave a message about partnering with my senator on this."
        ),
        "why_them": why,
        "the_ask": (
            "Ask your senator to partner with this colleague on kei truck legislation \u2014 "
            "they sit next to each other and can work together on the bill."
        ),
        "closing": "Thank you for your time.",
    }


def build_script_sections_super_ally(card: dict) -> dict:
    wow = script_wow_line(card)
    return {
        "opening": (
            "Hi, I\u2019m calling about kei truck legislation. "
            "I\u2019d like to leave a message about co-sponsorship and partnering with my senator."
        ),
        "why_them": (
            "This legislator is both the top influence in the chamber and sits next to your senator. "
            "Their support can move the bill in committee and on the floor." + wow
        ),
        "the_ask": (
            "Ask them to co-sponsor and to work with your senator. "
            "Constituent pressure plus their influence can unlock the process."
        ),
        "closing": "Thank you for your time.",
    }


def build_script_hint_senator(card: dict, zip_code: str, district: str) -> str:
    s = build_script_sections_senator(card, zip_code, district)
    return " ".join(filter(None, [s["opening"], s["why_them"], s["the_ask"], s["closing"]]))


def build_script_hint_rep(card: dict, zip_code: str, district: str) -> str:
    s = build_script_sections_rep(card, zip_code, district)
    return " ".join(filter(None, [s["opening"], s["why_them"], s["the_ask"], s["closing"]]))


def build_script_hint_broker(card: dict, broker_why: str) -> str:
    s = build_script_sections_broker(card, broker_why)
    return " ".join(filter(None, [s["opening"], s["why_them"], s["the_ask"], s["closing"]]))


def build_script_hint_ally(card: dict) -> str:
    s = build_script_sections_ally(card)
    return " ".join(filter(None, [s["opening"], s["why_them"], s["the_ask"], s["closing"]]))


def build_script_hint_super_ally(card: dict) -> str:
    s = build_script_sections_super_ally(card)
    return " ".join(filter(None, [s["opening"], s["why_them"], s["the_ask"], s["closing"]]))


def build_email_subject(zip_code: str) -> str:
    """Legacy: card prefill subject. Uses default poignant subject."""
    return build_email_subject_line(zip_code, variant="default")


def build_email_subject_line(zip_code: str, variant: str = "default") -> str:
    """Dynamic subject line. variant: default | alternate_short | alternate_direct."""
    zip_ = zip_code or "____"
    if variant == "alternate_short":
        return f'Please support a narrow "kei vehicle" registration fix ({zip_} constituent)'
    if variant == "alternate_direct":
        return "Will [LEGISLATOR_TITLE] [LEGISLATOR_LAST] support kei vehicle legislation?"
    return "Constituent request: clarify Illinois registration for kei vehicles (25+ years old)"


def build_email_ask_block(title_label: str, legislator_last: str, target_type: str) -> str:
    """Dynamic ASK block for email body (narrow bill, 25+ year kei, limited on-road)."""
    title = title_label or "[LEGISLATOR_TITLE]"
    last = legislator_last or "[LEGISLATOR_LAST]"
    return (
        f"My ask: Please support and help advance a narrow, safety-focused bill that explicitly "
        f"authorizes registration of 25+ year old kei vehicles for limited on-road use "
        f"(e.g., no interstates, equipment and insurance requirements). I'm asking {title} {last} "
        "to support and help advance this fix."
    )


def _time_of_day_greeting() -> str:
    """Return 'morning', 'afternoon', or 'evening' based on current hour (server local time)."""
    hour = datetime.now().hour
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    return "evening"


def build_email_body_full(
    *,
    legislator_title: str,
    legislator_last: str,
    legislator_full: str,
    office_name: str,
    district: str,
    zip_code: str,
    target_type: str,
    called_or_email_first: str,
    staffer_name: str = "",
    call_date: str = "",
    one_pager_note: str = "Attached is a two-page brief with the core argument and optional supporting detail.",
) -> str:
    """Dynamic email body: what's happening, why (3-401(c-1)), my ask (bullets), next step.

    Placeholders for user: [CALLER_NAME], [CALLER_CITY], [CALLER_ZIP], [CALLER_PHONE], [CALLER_EMAIL].
    Greeting: if staffer_name (person they talked to on the phone), use "Good [time], [Name],";
    otherwise use "Dear [Office of Rep./Sen. Last]," for a more official tone.
    """
    if staffer_name:
        time_word = _time_of_day_greeting()
        greeting = f"Good {time_word}, {staffer_name},\n\n"
    else:
        greeting = f"Dear {office_name},\n\n"
    intro = "My name is [CALLER_NAME] and I'm a constituent in [CALLER_CITY] [CALLER_ZIP]. "
    if called_or_email_first == "CALLED":
        intro += (
            f"I'm following up on my call to request {legislator_title} {legislator_last}'s "
            "support on a narrow legislative fix for kei vehicles in Illinois.\n\n"
        )
    else:
        intro += (
            f"I'm reaching out to request {legislator_title} {legislator_last}'s support on a "
            "narrow legislative fix for kei vehicles in Illinois.\n\n"
        )
    whats_happening = (
        "What's happening: The Illinois Secretary of State is treating kei vehicles as "
        'off-highway/non-highway and branding titles "Not Eligible for Registration," stating '
        "that states which allow on-road use do so by statute.\n\n"
    )
    why_plain = (
        "Why (in plain terms): Illinois is relying on the registration eligibility test in "
        '625 ILCS 5/3-401(c-1) (the "originally manufactured for highway use" requirement) to '
        "deny registration.\n\n"
    )
    my_ask = (
        "My ask: Please support and help advance a narrow, safety-focused bill that:\n"
        '• Defines a "Kei Vehicle" category (e.g., 25+ years old)\n'
        "• Authorizes registration for limited on-road use with clear restrictions (e.g., no interstates)\n"
        "• Sets standard requirements (equipment, insurance, documentation)\n\n"
    )
    closing = (
        f"{one_pager_note}\n\n"
        "[NEXT_STEP_ASK]\n\n"
        "Sincerely,\n"
        "[CALLER_NAME]\n"
        "[CALLER_CITY], IL [CALLER_ZIP]\n"
        "[CALLER_PHONE]\n"
    )
    return greeting + intro + whats_happening + why_plain + my_ask + closing


def build_email_body(member_name: str, script_hint: str, has_public_email: bool) -> str:
    """Legacy: card prefill body. Uses generic script_hint paragraph."""
    body = (
        f"Dear {member_name},\n\n"
        "I am a constituent writing to urge your support for kei truck legalization in Illinois.\n\n"
        f"{script_hint}\n\n"
        "Thank you for your service and your consideration.\n\n"
        "Sincerely,\n"
        "[YOUR FULL NAME]\n"
        "[YOUR STREET ADDRESS]\n"
        "[YOUR CITY, STATE ZIP]\n"
        "[YOUR PHONE]\n"
        "[OPTIONAL: One sentence on why this matters to you]\n"
    )
    if not has_public_email:
        body = (
            "(If this legislator\u2019s email is not publicly listed, call their office and ask for "
            "the best email address for constituent correspondence. Use the Call script to get the number.)\n\n"
            + body
        )
    return body


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


def build_after_call_email_subject(zip_code: str) -> str:
    return build_email_subject_line(zip_code, variant="default")


def build_after_call_email_body(
    staffer_name: str,
    legislator_name: str,
    zip_code: str,
    *,
    chamber: str | None = None,
    district: str | None = None,
    target_type: str = "NON_COMMITTEE",
    call_date: str = "",
) -> str:
    """After-call follow-up email body with dynamic ASK block (POWER_BROKER vs NON_COMMITTEE)."""
    title_label, legislator_last, legislator_full, office_name, district_label = (
        _legislator_email_context(legislator_name, chamber, district)
    )
    return build_email_body_full(
        legislator_title=title_label,
        legislator_last=legislator_last,
        legislator_full=legislator_full,
        office_name=office_name,
        district=district_label,
        zip_code=zip_code or "",
        target_type="POWER_BROKER" if target_type == "POWER_BROKER" else "NON_COMMITTEE",
        called_or_email_first="CALLED",
        staffer_name=staffer_name or "",
        call_date=call_date,
    )


def build_email_first_subject(zip_code: str) -> str:
    return build_email_subject_line(zip_code, variant="default")


def build_email_first_body(
    legislator_name: str,
    zip_code: str,
    *,
    chamber: str | None = None,
    district: str | None = None,
    target_type: str = "NON_COMMITTEE",
) -> str:
    """Email-first (no prior call) body with dynamic ASK block (POWER_BROKER vs NON_COMMITTEE)."""
    title_label, legislator_last, legislator_full, office_name, district_label = (
        _legislator_email_context(legislator_name, chamber, district)
    )
    return build_email_body_full(
        legislator_title=title_label,
        legislator_last=legislator_last,
        legislator_full=legislator_full,
        office_name=office_name,
        district=district_label,
        zip_code=zip_code or "",
        target_type="POWER_BROKER" if target_type == "POWER_BROKER" else "NON_COMMITTEE",
        called_or_email_first="EMAIL_FIRST",
        staffer_name="",
        call_date="",
    )


def find_power_broker(
    state: Any,
    exclude_district: str,
    *,
    committee_ids: set[str] | None = None,
    committee_codes: list[str] | None = None,
    category_name: str = "",
) -> tuple[Member | None, str]:
    """Find the top Senate Power Broker. Returns (Member | None, why_text)."""
    member_lookup = {m.id: m for m in state.members}

    if committee_codes:
        for code in committee_codes:
            for cmr in state.committee_rosters.get(code, []):
                role_lower = cmr.role.lower()
                if "chair" in role_lower and "vice" not in role_lower:
                    chair_member = member_lookup.get(cmr.member_id)
                    if (
                        chair_member
                        and chair_member.chamber == "Senate"
                        and chair_member.district != exclude_district
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
        if profile.chamber != "Senate":
            continue
        if profile.district == exclude_district:
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

    parts = [
        f"Highest Moneyball score ({best_profile.moneyball_score}) "
        f"in the Senate outside your district",
    ]
    if category_name:
        parts.append(f"sits on a {category_name} committee")
    parts.append(
        f"effectiveness: {best_profile.effectiveness_rate:.0%}, "
        f"{best_profile.unique_collaborators} collaborators"
    )
    why = ". ".join(parts) + "."
    return member, why


def find_ally(
    state: Any,
    senator: Member,
    *,
    committee_ids: set[str] | None = None,
    category_name: str = "",
) -> tuple[Member | None, str]:
    """Find the best Ally from the senator's seatmates. Returns (Member | None, why_text)."""
    if not senator.seatmate_names:
        return None, ""

    best_member = None
    best_bridge = -1.0

    for seatmate_name in senator.seatmate_names:
        member = state.member_lookup.get(seatmate_name)
        if member is None:
            continue
        if committee_ids and member.id not in committee_ids:
            continue

        bridge = 0.0
        if state.moneyball:
            mb = state.moneyball.profiles.get(member.id)
            if mb:
                bridge = mb.bridge_score

        if bridge > best_bridge:
            best_bridge = bridge
            best_member = member

    if best_member is None and committee_ids:
        return find_ally(state, senator, committee_ids=None, category_name="")

    if best_member is None:
        for seatmate_name in senator.seatmate_names:
            member = state.member_lookup.get(seatmate_name)
            if member is not None:
                best_member = member
                break

    if best_member is None:
        return None, ""

    why_parts = ["Sits next to your senator in the chamber"]
    if category_name and committee_ids and best_member.id in committee_ids:
        why_parts.append(f"also on a {category_name} committee")
    if state.moneyball:
        mb = state.moneyball.profiles.get(best_member.id)
        if mb and mb.bridge_score > 0:
            why_parts.append(
                f"bridge score of {mb.bridge_score:.0%} (cross-party co-sponsorship rate)"
            )
    if senator.seatmate_affinity > 0:
        why_parts.append(f"{senator.seatmate_affinity:.0%} bill overlap with seatmates")
    why = ". ".join(why_parts) + "."
    return best_member, why


def test_member_list(state: Any, max_count: int = 20) -> list[dict[str, Any]]:
    """First N members for test-mode dropdown/jump links."""
    out: list[dict[str, Any]] = []
    for m in state.members[:max_count]:
        out.append({"id": m.id, "name": m.name})
    return out
