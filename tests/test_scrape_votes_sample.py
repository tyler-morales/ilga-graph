"""Tests for the sample-based vote/slip scraping strategy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ilga_graph.models import Bill

# ── Mock data fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def mock_bills_cache(tmp_path: Path) -> dict[str, Bill]:
    """Generate 100 mock bills for testing sample strategy."""
    bills = {}
    for i in range(1, 101):
        bill_num = f"HB{i:04d}"
        leg_id = f"{200000 + i}"
        bills[leg_id] = Bill(
            bill_number=bill_num,
            leg_id=leg_id,
            description=f"Test bill {bill_num} with realistic description text for testing",
            chamber="H",
            last_action="In Committee" if i % 3 == 0 else "Third Reading",
            last_action_date=f"{i % 12 + 1}/1/2025",
            primary_sponsor="Test Sponsor",
            status_url=f"https://www.ilga.gov/legislation/BillStatus.asp?DocNum={i}&DocTypeID=HB",
        )
    return bills


@pytest.fixture
def mock_vote_events() -> list[dict[str, Any]]:
    """Mock vote event data for testing."""
    return [
        {
            "bill_number": "HB0010",
            "vote_type": "3rd Reading",
            "chamber": "H",
            "vote_date": "2025-03-15",
            "yeas": 75,
            "nays": 42,
            "present": 1,
            "description": "FINAL PASSAGE - SHORT DEBATE",
            "voters": {
                "Neil Anderson": "Y",
                "Omar Aquino": "Y",
                "Sue Rezin": "N",
            },
        },
        {
            "bill_number": "HB0020",
            "vote_type": "Committee",
            "chamber": "H",
            "vote_date": "2025-02-10",
            "yeas": 8,
            "nays": 3,
            "present": 0,
            "description": "Agriculture Committee - Do Pass",
            "voters": {
                "Neil Anderson": "Y",
                "Omar Aquino": "N",
            },
        },
        {
            "bill_number": "HB0030",
            "vote_type": "3rd Reading",
            "chamber": "H",
            "vote_date": "2025-04-20",
            "yeas": 90,
            "nays": 25,
            "present": 3,
            "description": "FINAL PASSAGE",
            "voters": {
                "Neil Anderson": "Y",
                "Omar Aquino": "Y",
                "Sue Rezin": "Y",
            },
        },
    ]


@pytest.fixture
def mock_witness_slips() -> list[dict[str, Any]]:
    """Mock witness slip data for testing."""
    return [
        {
            "bill_number": "HB0010",
            "name": "Jane Advocate",
            "organization": "Illinois Advocacy Group",
            "representing": "Illinois Advocacy Group",
            "position": "Proponent",
            "hearing_committee": "Executive",
            "hearing_date": "2025-03-01 14:00",
        },
        {
            "bill_number": "HB0010",
            "name": "Bob Opponent",
            "organization": "Citizens Against Change",
            "representing": "Citizens Against Change",
            "position": "Opponent",
            "hearing_committee": "Executive",
            "hearing_date": "2025-03-01 14:00",
        },
        {
            "bill_number": "HB0020",
            "name": "Alice Farmer",
            "organization": "Illinois Farm Bureau",
            "representing": "Illinois Farm Bureau",
            "position": "Proponent",
            "hearing_committee": "Agriculture",
            "hearing_date": "2025-02-01 10:00",
        },
    ]


# ── Sample strategy helpers ───────────────────────────────────────────────────


def get_sample_bill_numbers(all_bills: list[str], sample_rate: int = 10) -> list[str]:
    """Extract every Nth bill for sampling strategy.

    Args:
        all_bills: Sorted list of bill numbers (e.g., ["HB0001", "HB0002", ...])
        sample_rate: Take every Nth bill (default: 10 = 10% sample)

    Returns:
        Sampled bill numbers in original order
    """
    return [bill for i, bill in enumerate(all_bills) if i % sample_rate == 0]


def get_remaining_bills(
    all_bills: list[str],
    sampled_bills: set[str],
) -> list[str]:
    """Get bills that weren't in the sample (for gap-filling phase).

    Args:
        all_bills: All bill numbers
        sampled_bills: Bills already scraped in sample phase

    Returns:
        Remaining bills to scrape
    """
    return [b for b in all_bills if b not in sampled_bills]


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSampleStrategy:
    """Test the sample-first scraping strategy."""

    def test_sample_rate_10_percent(self, mock_bills_cache: dict[str, Bill]) -> None:
        """Sampling every 10th bill gives ~10% of total."""
        all_bills = sorted([b.bill_number for b in mock_bills_cache.values()])
        sample = get_sample_bill_numbers(all_bills, sample_rate=10)

        # 100 bills / 10 = 10 bills in sample
        assert len(sample) == 10
        assert sample[0] == "HB0001"  # First bill
        assert sample[1] == "HB0011"  # Every 10th
        assert sample[-1] == "HB0091"

    def test_sample_rate_5_percent(self, mock_bills_cache: dict[str, Bill]) -> None:
        """Sampling every 20th bill gives ~5% of total."""
        all_bills = sorted([b.bill_number for b in mock_bills_cache.values()])
        sample = get_sample_bill_numbers(all_bills, sample_rate=20)

        # 100 bills / 20 = 5 bills in sample
        assert len(sample) == 5
        assert sample[0] == "HB0001"
        assert sample[1] == "HB0021"

    def test_remaining_bills_excludes_sample(
        self,
        mock_bills_cache: dict[str, Bill],
    ) -> None:
        """Gap-filling phase gets all non-sampled bills."""
        all_bills = sorted([b.bill_number for b in mock_bills_cache.values()])
        sample = get_sample_bill_numbers(all_bills, sample_rate=10)
        remaining = get_remaining_bills(all_bills, set(sample))

        # 100 total - 10 sampled = 90 remaining
        assert len(remaining) == 90
        assert "HB0001" not in remaining  # Was in sample
        assert "HB0002" in remaining  # Was not in sample
        assert "HB0011" not in remaining  # Was in sample
        assert "HB0012" in remaining

    def test_no_overlap_between_sample_and_remaining(
        self,
        mock_bills_cache: dict[str, Bill],
    ) -> None:
        """Sample and remaining sets are disjoint."""
        all_bills = sorted([b.bill_number for b in mock_bills_cache.values()])
        sample = get_sample_bill_numbers(all_bills, sample_rate=10)
        remaining = get_remaining_bills(all_bills, set(sample))

        assert len(set(sample) & set(remaining)) == 0

    def test_sample_plus_remaining_equals_total(
        self,
        mock_bills_cache: dict[str, Bill],
    ) -> None:
        """Sample + remaining = all bills."""
        all_bills = sorted([b.bill_number for b in mock_bills_cache.values()])
        sample = get_sample_bill_numbers(all_bills, sample_rate=10)
        remaining = get_remaining_bills(all_bills, set(sample))

        assert len(sample) + len(remaining) == len(all_bills)
        assert set(sample) | set(remaining) == set(all_bills)


class TestMockData:
    """Verify mock data structure matches real scraped data."""

    def test_mock_bills_structure(self, mock_bills_cache: dict[str, Bill]) -> None:
        """Mock bills have required fields."""
        bill = list(mock_bills_cache.values())[0]
        assert bill.bill_number.startswith("HB")
        assert bill.leg_id
        assert bill.description
        assert bill.chamber == "H"
        assert bill.last_action
        assert bill.last_action_date
        assert bill.primary_sponsor
        assert bill.status_url

    def test_mock_vote_events_structure(
        self,
        mock_vote_events: list[dict[str, Any]],
    ) -> None:
        """Mock vote events have required fields."""
        vote = mock_vote_events[0]
        assert "bill_number" in vote
        assert "vote_type" in vote
        assert "chamber" in vote
        assert "vote_date" in vote
        assert "yeas" in vote
        assert "nays" in vote
        assert "voters" in vote
        assert isinstance(vote["voters"], dict)

    def test_mock_witness_slips_structure(
        self,
        mock_witness_slips: list[dict[str, Any]],
    ) -> None:
        """Mock witness slips have required fields."""
        slip = mock_witness_slips[0]
        assert "bill_number" in slip
        assert "name" in slip
        assert "organization" in slip
        assert "position" in slip
        assert "hearing_committee" in slip
        assert "hearing_date" in slip


class TestSampleStrategyIntegration:
    """Integration tests for sample strategy with realistic scenarios."""

    def test_11721_bills_sample_10_percent(self) -> None:
        """Real dataset: 11,721 bills → 1,173 in 10% sample (includes index 0)."""
        all_bills = [f"HB{i:04d}" for i in range(1, 11722)]
        sample = get_sample_bill_numbers(all_bills, sample_rate=10)

        # Sampling indices 0, 10, 20, ... 11720 = 1173 bills (includes first bill)
        assert len(sample) == 1173

    def test_sample_preserves_bill_order(self) -> None:
        """Sample maintains sequential order (HB0001, HB0011, HB0021...)."""
        all_bills = [f"HB{i:04d}" for i in range(1, 101)]
        sample = get_sample_bill_numbers(all_bills, sample_rate=10)

        # Check that sample is sorted
        assert sample == sorted(sample)

        # Check spacing
        for i in range(len(sample) - 1):
            # Extract numeric part
            curr_num = int(sample[i].replace("HB", ""))
            next_num = int(sample[i + 1].replace("HB", ""))
            # Should be spaced by 10
            assert next_num - curr_num == 10

    def test_time_estimate_for_sample(self) -> None:
        """Calculate realistic time estimates for 10% sample."""
        total_bills = 11721
        sample_rate = 10
        sample_size = total_bills // sample_rate
        avg_time_per_bill_seconds = 3.7

        estimated_time_seconds = sample_size * avg_time_per_bill_seconds
        estimated_time_minutes = estimated_time_seconds / 60

        # ~1172 bills * 3.7s = 4336s = 72 minutes
        assert sample_size == 1172
        assert 70 <= estimated_time_minutes <= 75


# ── Progress tracking tests ───────────────────────────────────────────────────


class TestProgressTracking:
    """Test progress file format for sample strategy."""

    def test_progress_file_tracks_sample_phase(self, tmp_path: Path) -> None:
        """Progress file includes sample_phase flag."""
        progress_file = tmp_path / "votes_slips_progress.json"

        progress_data = {
            "scraped_bill_numbers": ["HB0001", "HB0011", "HB0021"],
            "sample_phase": True,
            "sample_rate": 10,
            "updated_at": "2025-02-11T00:00:00Z",
        }

        with open(progress_file, "w") as f:
            json.dump(progress_data, f)

        # Read back
        with open(progress_file) as f:
            loaded = json.load(f)

        assert loaded["sample_phase"] is True
        assert loaded["sample_rate"] == 10
        assert len(loaded["scraped_bill_numbers"]) == 3

    def test_progress_transitions_to_gap_fill(self, tmp_path: Path) -> None:
        """Progress file transitions from sample to gap-fill phase."""
        progress_file = tmp_path / "votes_slips_progress.json"

        # Sample phase complete
        sample_complete = {
            "scraped_bill_numbers": ["HB0001", "HB0011"],
            "sample_phase": True,
            "sample_complete": True,
            "sample_rate": 10,
            "updated_at": "2025-02-11T00:00:00Z",
        }

        with open(progress_file, "w") as f:
            json.dump(sample_complete, f)

        # Now start gap fill
        with open(progress_file) as f:
            loaded = json.load(f)

        assert loaded["sample_complete"] is True

        # Add gap-fill bills
        gap_fill = loaded.copy()
        gap_fill["scraped_bill_numbers"].extend(["HB0002", "HB0003"])
        gap_fill["gap_fill_phase"] = True

        with open(progress_file, "w") as f:
            json.dump(gap_fill, f)

        with open(progress_file) as f:
            final = json.load(f)

        assert final["gap_fill_phase"] is True
        assert len(final["scraped_bill_numbers"]) == 4
