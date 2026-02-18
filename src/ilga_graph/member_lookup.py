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
