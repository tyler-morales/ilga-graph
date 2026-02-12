"""Request-scoped batch loaders for GraphQL resolvers.

Provides batch_load(keys) for scorecards, moneyball profiles, bills, and members
so resolvers can fetch related data in one go instead of N lookups.
"""

from __future__ import annotations

from typing import Any

from .analytics import MemberScorecard
from .models import Bill, Member
from .moneyball import MoneyballProfile


class ScorecardLoader:
    """Batch loader for member_id -> MemberScorecard."""

    def __init__(self, scorecards: dict[str, MemberScorecard]) -> None:
        self._scorecards = scorecards

    def load(self, member_id: str) -> MemberScorecard | None:
        return self._scorecards.get(member_id)

    def batch_load(self, member_ids: list[str]) -> list[MemberScorecard | None]:
        return [self._scorecards.get(mid) for mid in member_ids]


class MoneyballProfileLoader:
    """Batch loader for member_id -> MoneyballProfile."""

    def __init__(self, moneyball: Any) -> None:
        self._profiles = moneyball.profiles if moneyball else {}

    def load(self, member_id: str) -> MoneyballProfile | None:
        return self._profiles.get(member_id)

    def batch_load(self, member_ids: list[str]) -> list[MoneyballProfile | None]:
        return [self._profiles.get(mid) for mid in member_ids]


class BillLoader:
    """Batch loader for bill_number -> Bill."""

    def __init__(self, bill_lookup: dict[str, Bill]) -> None:
        self._bill_lookup = bill_lookup

    def load(self, bill_number: str) -> Bill | None:
        return self._bill_lookup.get(bill_number)

    def batch_load(self, bill_numbers: list[str]) -> list[Bill | None]:
        return [self._bill_lookup.get(bn) for bn in bill_numbers]


class MemberLoader:
    """Batch loader for member_id -> Member."""

    def __init__(self, member_lookup: dict[str, Member]) -> None:
        self._member_lookup = member_lookup

    def load(self, member_id: str) -> Member | None:
        return self._member_lookup.get(member_id)

    def batch_load(self, member_ids: list[str]) -> list[Member | None]:
        return [self._member_lookup.get(mid) for mid in member_ids]


def create_loaders(state: Any) -> dict[str, Any]:
    """Create request-scoped loaders from app state."""
    return {
        "state": state,
        "scorecard_loader": ScorecardLoader(state.scorecards),
        "moneyball_loader": MoneyballProfileLoader(state.moneyball),
        "bill_loader": BillLoader(state.bill_lookup),
        "member_loader": MemberLoader(
            {m.id: m for m in state.members},
        ),
    }
