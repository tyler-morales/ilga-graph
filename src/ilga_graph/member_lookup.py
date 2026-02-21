"""Member lookup helpers. Used by advocacy, explore, and intelligence."""

from typing import TYPE_CHECKING

from .models import Member

if TYPE_CHECKING:
    from .app_state import AppState


def find_member_by_id(state: "AppState", member_id: str) -> Member | None:
    """Find a member by id."""
    return state.member_lookup_by_id.get(member_id) if member_id else None


def find_member_by_district(state: "AppState", chamber: str, district: str) -> Member | None:
    """Find a member by chamber (case-insensitive) and district number."""
    chamber_lower = chamber.lower()
    for m in state.members:
        if m.chamber.lower() == chamber_lower and m.district == district:
            return m
    return None


def is_constituent_for_zip_member(state: "AppState", zip_code: str, member: Member | None) -> bool:
    """True if member is the senator or representative for the given ZIP's district."""
    if not zip_code or not member:
        return False
    district_info = state.zip_to_district.get(zip_code)
    if not district_info:
        return False
    senator = (
        find_member_by_district(state, "senate", district_info.il_senate)
        if district_info.il_senate
        else None
    )
    rep = (
        find_member_by_district(state, "house", district_info.il_house)
        if district_info.il_house
        else None
    )
    return (senator and member.id == senator.id) or (rep and member.id == rep.id)
