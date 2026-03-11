"""Tests for common reports scraper (passed both houses, pending governor)."""

from __future__ import annotations

from unittest.mock import MagicMock

from ilga_graph.scrapers.reports import (
    reports_to_bill_numbers,
    scrape_common_reports,
)


class TestReportsToBillNumbers:
    def test_returns_set(self) -> None:
        assert reports_to_bill_numbers(["SB0001", "HB0042"]) == {"SB0001", "HB0042"}

    def test_dedupes(self) -> None:
        assert reports_to_bill_numbers(["SB0001", "SB0001"]) == {"SB0001"}

    def test_empty(self) -> None:
        assert reports_to_bill_numbers([]) == set()


class TestScrapeCommonReports:
    def test_returns_list_when_session_returns_empty_html(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "<html><body></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp
        out = scrape_common_reports(session=mock_session, request_delay=0)
        assert isinstance(out, list)
        assert out == []

    def test_extracts_bill_numbers_from_table_text(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = (
            "<html><body><table><tr><td>SB1234</td><td>Title</td></tr>"
            "<tr><td>HB5678</td></tr></table></body></html>"
        )
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp
        out = scrape_common_reports(session=mock_session, request_delay=0)
        assert "SB1234" in out
        assert "HB5678" in out
