"""Mapping service: hydrates raw app state into ontology objects with links."""

from __future__ import annotations

from typing import Any

from ..app_state import AppState
from .links import LinkType, ObjectLink
from .objects import (
    BillObject,
    CommitteeObject,
    HearingObject,
    LegislatorObject,
    OrganizationObject,
    VoteEventObject,
)


def _member_id_from_name(state: AppState, name: str) -> str | None:
    m = state.member_lookup.get(name) if name else None
    return m.id if m else None


def _leg_id_from_bill_number(state: AppState, bill_number: str) -> str | None:
    b = state.bill_lookup.get(bill_number) if bill_number else None
    return b.leg_id if b else None


def _career_ranges_to_dicts(member: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cr in getattr(member, "career_ranges", []) or []:
        out.append(
            {
                "start_year": getattr(cr, "start_year", 0),
                "end_year": getattr(cr, "end_year"),
                "chamber": getattr(cr, "chamber"),
            }
        )
    return out


def _scorecard_to_dict(sc: Any) -> dict[str, Any] | None:
    if sc is None:
        return None
    return {
        "primary_bill_count": getattr(sc, "primary_bill_count", 0),
        "passed_count": getattr(sc, "passed_count", 0),
        "success_rate": getattr(sc, "success_rate", 0.0),
        "heat_score": getattr(sc, "heat_score", 0),
        "effectiveness_score": getattr(sc, "effectiveness_score", 0.0),
        "magnet_score": getattr(sc, "magnet_score", 0.0),
        "bridge_score": getattr(sc, "bridge_score", 0.0),
    }


def _moneyball_to_dict(mb: Any) -> dict[str, Any] | None:
    if mb is None:
        return None
    return {
        "moneyball_score": getattr(mb, "moneyball_score", 0.0),
        "laws_filed": getattr(mb, "laws_filed", 0),
        "effectiveness_rate": getattr(mb, "effectiveness_rate", 0.0),
        "network_centrality": getattr(mb, "network_centrality", 0.0),
        "rank_overall": getattr(mb, "rank_overall", 0),
        "rank_chamber": getattr(mb, "rank_chamber", 0),
    }


def _influence_to_dict(inf: Any) -> dict[str, Any] | None:
    if inf is None:
        return None
    return {
        "influence_score": getattr(inf, "influence_score", 0.0),
        "influence_label": getattr(inf, "influence_label", ""),
        "rank_overall": getattr(inf, "rank_overall", 0),
        "rank_chamber": getattr(inf, "rank_chamber", 0),
        "influence_signals": getattr(inf, "influence_signals", []),
    }


class MappingService:
    """Converts raw SQL/cache data into ontology objects with resolved links."""

    def __init__(self, state: AppState) -> None:
        self._state = state

    def hydrate_legislator(self, member_id: str) -> LegislatorObject | None:
        member = self._state.member_lookup_by_id.get(member_id)
        if not member:
            return None
        links: list[ObjectLink] = []
        for leg_id in getattr(member, "sponsored_bill_ids", []) or []:
            links.append(
                ObjectLink(
                    target_id=leg_id,
                    target_type="bill",
                    link_type=LinkType.SPONSORS.value,
                )
            )
        for leg_id in getattr(member, "co_sponsor_bill_ids", []) or []:
            links.append(
                ObjectLink(
                    target_id=leg_id,
                    target_type="bill",
                    link_type=LinkType.CO_SPONSORS.value,
                )
            )
        for code, roster in self._state.committee_rosters.items():
            for cmr in roster or []:
                if getattr(cmr, "member_id", None) == member_id:
                    role = getattr(cmr, "role", "") or ""
                    links.append(
                        ObjectLink(
                            target_id=code,
                            target_type="committee",
                            link_type=LinkType.COMMITTEE_MEMBERSHIP.value,
                            metadata={"role": role},
                        )
                    )
                    break
        scorecard = _scorecard_to_dict(self._state.scorecards.get(member_id))
        moneyball = None
        if self._state.moneyball and self._state.moneyball.profiles:
            moneyball = _moneyball_to_dict(self._state.moneyball.profiles.get(member_id))
        influence = _influence_to_dict(self._state.influence.get(member_id))
        return LegislatorObject(
            object_id=member_id,
            name=member.name,
            district=getattr(member, "district", "") or "",
            party=getattr(member, "party", "") or "",
            chamber=getattr(member, "chamber", "") or "",
            role=getattr(member, "role", "") or "",
            photo_url=getattr(member, "photo_url", "") or "",
            career_ranges=_career_ranges_to_dicts(member),
            links=links,
            scorecard=scorecard,
            moneyball=moneyball,
            influence=influence,
        )

    def hydrate_bill(self, leg_id: str) -> BillObject | None:
        bill = self._state.bills_lookup.get(leg_id)
        if not bill:
            return None
        links: list[ObjectLink] = []
        for mid in getattr(bill, "sponsor_ids", []) or []:
            links.append(
                ObjectLink(
                    target_id=mid,
                    target_type="legislator",
                    link_type=LinkType.SPONSORED_BY.value,
                )
            )
        sponsor_ids = [lnk.target_id for lnk in links if lnk.link_type == LinkType.SPONSORED_BY.value]
        for mid in getattr(bill, "house_sponsor_ids", []) or []:
            if mid and mid not in sponsor_ids:
                links.append(
                    ObjectLink(
                        target_id=mid,
                        target_type="legislator",
                        link_type=LinkType.CO_SPONSORED_BY.value,
                    )
                )
        for code, bill_numbers in self._state.committee_bills.items():
            if bill.bill_number in (bill_numbers or []):
                links.append(
                    ObjectLink(
                        target_id=code,
                        target_type="committee",
                        link_type=LinkType.IN_COMMITTEE.value,
                    )
                )
        for i, ve in enumerate(getattr(bill, "vote_events", []) or []):
            vid = (
                f"vote|{bill.bill_number}|{getattr(ve, 'date', '')}"
                f"|{getattr(ve, 'chamber', '')}|{i}"
            )
            links.append(
                ObjectLink(
                    target_id=vid,
                    target_type="vote_event",
                    link_type=LinkType.VOTE_EVENTS.value,
                )
            )
        for slip in self._state.witness_slips_lookup.get(bill.bill_number, []) or []:
            slip_id = (
                f"slip|{bill.bill_number}|{getattr(slip, 'name', '')}"
                f"|{getattr(slip, 'hearing_date', '')}"
            )
            links.append(
                ObjectLink(
                    target_id=slip_id,
                    target_type="witness_slip",
                    link_type=LinkType.WITNESS_SLIPS.value,
                )
            )
        for h in self._state.hearings_by_bill.get(bill.bill_number, []) or []:
            hid = (
                f"hearing|{getattr(h, 'committee_id', '')}|{getattr(h, 'date', '')}"
                f"|{getattr(h, 'time', '')}"
            )
            links.append(
                ObjectLink(
                    target_id=hid,
                    target_type="hearing",
                    link_type=LinkType.HEARINGS.value,
                )
            )
        prediction = None
        if getattr(self._state, "ml", None) and hasattr(self._state.ml, "bill_scores"):
            for s in getattr(self._state.ml, "bill_scores", []) or []:
                bid = getattr(s, "bill_id", None) or getattr(s, "bill_number", None)
                if bid in (leg_id, bill.bill_number):
                    prediction = {
                        "prob_advance": getattr(s, "prob_advance", None),
                        "prob_law": getattr(s, "prob_law", None),
                        "predicted_outcome": getattr(s, "predicted_outcome", None),
                        "confidence": getattr(s, "confidence", None),
                    }
                    break
        controversy = None
        if hasattr(self._state, "witness_slips_lookup"):
            slips = self._state.witness_slips_lookup.get(bill.bill_number, []) or []
            if slips:
                pro = sum(
                    1 for s in slips
                    if (getattr(s, "position", "") or "").lower() == "proponent"
                )
                opp = sum(
                    1 for s in slips
                    if (getattr(s, "position", "") or "").lower() == "opponent"
                )
                total = pro + opp
                controversy = (opp / total) if total else 0.0
        return BillObject(
            object_id=leg_id,
            bill_number=bill.bill_number,
            leg_id=bill.leg_id,
            description=getattr(bill, "description", "") or "",
            chamber=getattr(bill, "chamber", "") or "",
            status="",
            synopsis=getattr(bill, "synopsis", "") or "",
            last_action=getattr(bill, "last_action", "") or "",
            last_action_date=getattr(bill, "last_action_date", "") or "",
            primary_sponsor=getattr(bill, "primary_sponsor", "") or "",
            links=links,
            prediction=prediction,
            controversy_score=controversy,
        )

    def hydrate_committee(self, code: str) -> CommitteeObject | None:
        committee = self._state.committee_lookup.get(code)
        if not committee:
            return None
        links: list[ObjectLink] = []
        for cmr in self._state.committee_rosters.get(code, []) or []:
            mid = getattr(cmr, "member_id", None)
            if mid:
                links.append(
                    ObjectLink(
                        target_id=mid,
                        target_type="legislator",
                        link_type=LinkType.MEMBERS.value,
                        metadata={"role": getattr(cmr, "role", "") or ""},
                    )
                )
        for bill_number in self._state.committee_bills.get(code, []) or []:
            leg_id = _leg_id_from_bill_number(self._state, bill_number)
            if leg_id:
                links.append(
                    ObjectLink(
                        target_id=leg_id,
                        target_type="bill",
                        link_type=LinkType.BILLS.value,
                    )
                )
        for h in self._state.hearings or []:
            if getattr(h, "committee_id", None) == code:
                hid = f"hearing|{code}|{getattr(h, 'date', '')}|{getattr(h, 'time', '')}"
                links.append(
                    ObjectLink(
                        target_id=hid,
                        target_type="hearing",
                        link_type=LinkType.COMMITTEE_HEARINGS.value,
                    )
                )
        return CommitteeObject(
            object_id=code,
            code=code,
            name=getattr(committee, "name", "") or code,
            links=links,
        )

    def hydrate_organization(self, canonical_name: str) -> OrganizationObject | None:
        from ..intelligence_helpers import canonical_organization_name

        links: list[ObjectLink] = []
        slip_count = 0
        legislator_ids: set[str] = set()
        for slip in self._state.witness_slips or []:
            raw_org = getattr(slip, "organization", "") or ""
            if canonical_organization_name(raw_org) != canonical_name:
                continue
            slip_count += 1
            bill_number = getattr(slip, "bill_number", "") or ""
            if bill_number:
                bill = self._state.bill_lookup.get(bill_number)
                if bill and getattr(bill, "primary_sponsor", ""):
                    mid = _member_id_from_name(self._state, bill.primary_sponsor)
                    if mid:
                        legislator_ids.add(mid)
        for mid in legislator_ids:
            links.append(
                ObjectLink(
                    target_id=mid,
                    target_type="legislator",
                    link_type=LinkType.LEGISLATORS_ALIGNED.value,
                )
            )
        return OrganizationObject(
            object_id=canonical_name,
            name=canonical_name,
            canonical_name=canonical_name,
            slip_count=slip_count,
            links=links,
        )

    def hydrate_vote_event(
        self, bill_number: str, date: str, chamber: str, index: int
    ) -> VoteEventObject | None:
        events = self._state.vote_lookup.get(bill_number, []) or []
        for i, ve in enumerate(events):
            if i != index:
                continue
            if (getattr(ve, "date", "") or "") != date or (
                getattr(ve, "chamber", "") or ""
            ) != chamber:
                continue
            vid = f"vote|{bill_number}|{date}|{chamber}|{i}"
            links: list[ObjectLink] = []
            leg_id = _leg_id_from_bill_number(self._state, bill_number)
            if leg_id:
                links.append(
                    ObjectLink(target_id=leg_id, target_type="bill", link_type=LinkType.BILL.value)
                )
            for name in getattr(ve, "yea_votes", []) or []:
                mid = _member_id_from_name(self._state, name)
                if mid:
                    links.append(
                        ObjectLink(
                            target_id=mid,
                            target_type="legislator",
                            link_type=LinkType.YEA_VOTERS.value,
                        )
                    )
            for name in getattr(ve, "nay_votes", []) or []:
                mid = _member_id_from_name(self._state, name)
                if mid:
                    links.append(
                        ObjectLink(
                            target_id=mid,
                            target_type="legislator",
                            link_type=LinkType.NAY_VOTERS.value,
                        )
                    )
            return VoteEventObject(
                object_id=vid,
                bill_number=bill_number,
                date=date,
                description=getattr(ve, "description", "") or "",
                chamber=chamber,
                yea_count=len(getattr(ve, "yea_votes", []) or []),
                nay_count=len(getattr(ve, "nay_votes", []) or []),
                present_count=len(getattr(ve, "present_votes", []) or []),
                nv_count=len(getattr(ve, "nv_votes", []) or []),
                vote_type=getattr(ve, "vote_type", "floor") or "floor",
                links=links,
            )
        return None

    def hydrate_hearing(self, committee_id: str, date: str, time: str) -> HearingObject | None:
        for h in self._state.hearings or []:
            if (
                (getattr(h, "committee_id", "") or "") == committee_id
                and (getattr(h, "date", "") or "") == date
                and (getattr(h, "time", "") or "") == time
            ):
                hid = f"hearing|{committee_id}|{date}|{time}"
                links: list[ObjectLink] = []
                links.append(
                    ObjectLink(
                        target_id=committee_id,
                        target_type="committee",
                        link_type=LinkType.COMMITTEE.value,
                    )
                )
                for bill_number in getattr(h, "bills", []) or []:
                    leg_id = _leg_id_from_bill_number(self._state, bill_number)
                    if leg_id:
                        links.append(
                            ObjectLink(
                                target_id=leg_id,
                                target_type="bill",
                                link_type=LinkType.HEARING_BILLS.value,
                            )
                        )
                return HearingObject(
                    object_id=hid,
                    date=date,
                    time=time,
                    location=getattr(h, "location", "") or "",
                    committee_name=getattr(h, "committee_name", "") or "",
                    committee_id=committee_id,
                    posting_date=getattr(h, "posting_date", "") or "",
                    status=getattr(h, "status", "") or "",
                    chamber=getattr(h, "chamber", "") or "",
                    links=links,
                )
        return None

    def hydrate_all_legislators(self) -> list[LegislatorObject]:
        out: list[LegislatorObject] = []
        for m in self._state.members or []:
            obj = self.hydrate_legislator(m.id)
            if obj:
                out.append(obj)
        return out

    def hydrate_all_bills(self) -> list[BillObject]:
        out: list[BillObject] = []
        seen: set[str] = set()
        for b in self._state.bills or []:
            leg_id = getattr(b, "leg_id", None) or ""
            if leg_id and leg_id not in seen:
                seen.add(leg_id)
                obj = self.hydrate_bill(leg_id)
                if obj:
                    out.append(obj)
        return out
