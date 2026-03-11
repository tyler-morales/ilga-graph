"""Tests for hearing schedule scraper and bill-number extraction."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup

from ilga_graph.models import Hearing
from ilga_graph.scrapers.hearings import (
    _parse_schedule_table,
    extract_bill_numbers_from_text,
    hearings_to_bill_numbers,
    load_hearings_cache,
    save_hearings_cache,
)


class TestExtractBillNumbersFromText:
    def test_single_sb(self) -> None:
        assert extract_bill_numbers_from_text("Subject Matter On: SB4076: AOC") == ["SB4076"]

    def test_multiple_bills(self) -> None:
        text = "SB4076: AOC, COGFA, JCAR, SB4132: WCC, SB4116: GOMB"
        out = extract_bill_numbers_from_text(text)
        assert "SB4076" in out
        assert "SB4132" in out
        assert "SB4116" in out
        assert len(out) == 3

    def test_house_bills_with_space(self) -> None:
        text = "HB 5652, HB 5645, HB 5715"
        out = extract_bill_numbers_from_text(text)
        assert "HB5652" in out
        assert "HB5645" in out
        assert "HB5715" in out

    def test_normalized_format(self) -> None:
        assert extract_bill_numbers_from_text("SB 1") == ["SB0001"]
        assert extract_bill_numbers_from_text("HB 42") == ["HB0042"]

    def test_empty(self) -> None:
        assert extract_bill_numbers_from_text("") == []
        assert extract_bill_numbers_from_text("No bills here") == []


class TestHearingsToBillNumbers:
    def test_collects_all_bills(self) -> None:
        hearings = [
            Hearing(
                "2026-03-12",
                "9:00 AM",
                "400 Capitol",
                "Appropriations",
                "",
                ["SB4076", "SB4132"],
                "",
                "normal",
                "Senate",
            ),
            Hearing(
                "2026-03-12",
                "10:00 AM",
                "212 Capitol",
                "Judiciary",
                "",
                ["SB4076", "HB5652"],
                "",
                "normal",
                "House",
            ),
        ]
        out = hearings_to_bill_numbers(hearings)
        assert out == {"SB4076", "SB4132", "HB5652"}

    def test_empty(self) -> None:
        assert hearings_to_bill_numbers([]) == set()


class TestParseScheduleTable:
    def test_parses_table_with_bill_numbers(self) -> None:
        html = """
        <table>
        <tr><th>Time</th><th>Committee</th><th>Subject Matter</th></tr>
        <tr><td>3/12/2026 9:00 AM</td><td>Appropriations</td>
        <td>Subject Matter On: SB4076, SB4132</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        hearings = _parse_schedule_table(soup, "Senate")
        assert len(hearings) == 1
        assert hearings[0].chamber == "Senate"
        assert hearings[0].committee_name == "Appropriations"
        assert "SB4076" in hearings[0].bills
        assert "SB4132" in hearings[0].bills

    def test_empty_table(self) -> None:
        soup = BeautifulSoup("<table><tr><td>No header</td></tr></table>", "html.parser")
        assert _parse_schedule_table(soup, "House") == []


class TestHearingsCacheRoundtrip:
    def test_save_and_load(self) -> None:
        hearings = [
            Hearing(
                "2026-03-12",
                "9:00 AM",
                "400 Capitol",
                "Appropriations",
                "123",
                ["SB4076"],
                "3/1/2026",
                "normal",
                "Senate",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_file = tmp_path / "hearings.json"
            import ilga_graph.scrapers.hearings as hearings_mod

            orig_file = hearings_mod.HEARINGS_CACHE_FILE
            orig_dir = hearings_mod.CACHE_DIR
            hearings_mod.HEARINGS_CACHE_FILE = cache_file
            hearings_mod.CACHE_DIR = tmp_path
            try:
                save_hearings_cache(hearings)
                assert cache_file.exists()
                loaded = load_hearings_cache()
                assert loaded is not None
                assert len(loaded) == 1
                assert loaded[0].date == "2026-03-12"
                assert loaded[0].bills == ["SB4076"]
            finally:
                hearings_mod.HEARINGS_CACHE_FILE = orig_file
                hearings_mod.CACHE_DIR = orig_dir

    def test_load_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            import ilga_graph.scrapers.hearings as hearings_mod

            orig = hearings_mod.HEARINGS_CACHE_FILE
            hearings_mod.HEARINGS_CACHE_FILE = Path(tmp) / "nonexistent.json"
            try:
                assert load_hearings_cache() is None
            finally:
                hearings_mod.HEARINGS_CACHE_FILE = orig


class TestIncrementalSignalIntegration:
    """Signal-driven incremental scrape defaults and params."""

    def test_incremental_bill_scrape_has_signal_params_and_reduced_rescrape_days(self) -> None:
        from ilga_graph.scrapers.bills import incremental_bill_scrape

        sig = inspect.signature(incremental_bill_scrape)
        assert sig.parameters["rescrape_recent_days"].default == 14
        assert sig.parameters["use_hearing_signals"].default is True
        assert sig.parameters["use_report_signals"].default is True
