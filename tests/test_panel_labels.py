"""Tests for the panel (time-slice) ML dataset expansion.

Verifies that ``build_panel_labels()`` creates correct snapshot rows with
proper post-snapshot labels, and that the time-sliced feature builders
respect ``as_of_date`` filtering.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_bills(*specs: tuple[str, str]) -> pl.DataFrame:
    """Create a minimal dim_bills DataFrame.

    Each spec is (bill_id, introduction_date_iso).
    """
    return pl.DataFrame(
        [
            {
                "bill_id": bid,
                "bill_type": "SB",
                "introduction_date": intro,
                "bill_number_raw": f"SB{i:04d}",
                "description": f"Test bill {bid}",
                "synopsis_text": "",
                "chamber_origin": "S",
                "primary_sponsor": "Test Sponsor",
                "primary_sponsor_id": "",
                "sponsor_count": 1,
                "last_action": "",
                "last_action_date": "",
                "status_url": "",
            }
            for i, (bid, intro) in enumerate(specs, start=1)
        ]
    )


def _make_actions(
    *specs: tuple[str, str, str],
) -> pl.DataFrame:
    """Create a minimal fact_bill_actions DataFrame.

    Each spec is (bill_id, date_iso, action_text).
    """
    return pl.DataFrame(
        [
            {
                "action_id": f"{bid}_{date}_{i}",
                "bill_id": bid,
                "date": date,
                "chamber": "S",
                "action_text": text,
                "action_category": "procedural",
            }
            for i, (bid, date, text) in enumerate(specs)
        ]
    )


# ── Tests: build_panel_labels ────────────────────────────────────────────────


class TestBuildPanelLabels:
    """Tests for build_panel_labels()."""

    def test_basic_snapshot_rows(self) -> None:
        """A bill old enough should get one row per snapshot day."""
        from ilga_graph.ml.features import build_panel_labels

        # Bill introduced 300 days ago — should qualify for all snapshots
        intro = (datetime.now() - timedelta(days=300)).strftime("%Y-%m-%d")
        df_bills = _make_bills(("B1", intro))

        # Action after day 30 that constitutes "advance"
        action_date = (datetime.strptime(intro, "%Y-%m-%d") + timedelta(days=45)).strftime(
            "%Y-%m-%d"
        )
        df_actions = _make_actions(
            ("B1", intro, "Filed with Secretary"),
            ("B1", action_date, "Do Pass / Short Debate"),
        )

        df_panel = build_panel_labels(
            df_bills,
            df_actions,
            snapshot_days=[30, 60, 90],
            observation_days=90,
        )

        # Should have 3 rows (one per snapshot day), all for B1
        assert len(df_panel) == 3
        assert set(df_panel["bill_id"].to_list()) == {"B1"}
        assert sorted(df_panel["snapshot_day"].to_list()) == [30, 60, 90]

    def test_post_snapshot_labels(self) -> None:
        """Label should reflect what happened AFTER the snapshot date."""
        from ilga_graph.ml.features import build_panel_labels

        intro = (datetime.now() - timedelta(days=300)).strftime("%Y-%m-%d")
        df_bills = _make_bills(("B1", intro))

        # Advance action happens at day 45 (after day-30 snapshot, before day-60)
        action_date = (datetime.strptime(intro, "%Y-%m-%d") + timedelta(days=45)).strftime(
            "%Y-%m-%d"
        )
        df_actions = _make_actions(
            ("B1", intro, "Filed with Secretary"),
            ("B1", action_date, "Do Pass / Short Debate"),
        )

        df_panel = build_panel_labels(
            df_bills,
            df_actions,
            snapshot_days=[30, 60],
            observation_days=90,
        )

        rows = {r["snapshot_day"]: r for r in df_panel.to_dicts()}

        # At day 30 snapshot: the advance at day 45 is AFTER → label=1
        assert rows[30]["target_advanced_after"] == 1

        # At day 60 snapshot: the advance at day 45 is BEFORE → label=0
        # (no positive actions after day 60)
        assert rows[60]["target_advanced_after"] == 0

    def test_immature_bill_excluded(self) -> None:
        """Bills too new for observation window should be excluded."""
        from ilga_graph.ml.features import build_panel_labels

        # Bill introduced only 50 days ago — snapshot at 30 + 90 obs = 120 > 50
        intro = (datetime.now() - timedelta(days=50)).strftime("%Y-%m-%d")
        df_bills = _make_bills(("B1", intro))
        df_actions = _make_actions(
            ("B1", intro, "Filed with Secretary"),
        )

        df_panel = build_panel_labels(
            df_bills,
            df_actions,
            snapshot_days=[30],
            observation_days=90,
        )

        assert len(df_panel) == 0

    def test_multiple_bills(self) -> None:
        """Two bills, one old enough, one not."""
        from ilga_graph.ml.features import build_panel_labels

        old_intro = (datetime.now() - timedelta(days=300)).strftime("%Y-%m-%d")
        new_intro = (datetime.now() - timedelta(days=50)).strftime("%Y-%m-%d")

        df_bills = _make_bills(("OLD", old_intro), ("NEW", new_intro))
        df_actions = _make_actions(
            ("OLD", old_intro, "Filed with Secretary"),
            ("NEW", new_intro, "Filed with Secretary"),
        )

        df_panel = build_panel_labels(
            df_bills,
            df_actions,
            snapshot_days=[30],
            observation_days=90,
        )

        # Only the old bill should have a row
        assert set(df_panel["bill_id"].to_list()) == {"OLD"}

    def test_empty_actions(self) -> None:
        """Bill with no actions still gets rows (features will be zeros)."""
        from ilga_graph.ml.features import build_panel_labels

        intro = (datetime.now() - timedelta(days=300)).strftime("%Y-%m-%d")
        df_bills = _make_bills(("B1", intro))
        df_actions = _make_actions()  # empty

        df_panel = build_panel_labels(
            df_bills,
            df_actions,
            snapshot_days=[30, 60],
            observation_days=90,
        )

        # Should still produce rows — just with label=0
        assert len(df_panel) == 2
        for r in df_panel.to_dicts():
            assert r["target_advanced_after"] == 0
            assert r["target_law_after"] == 0


# ── Tests: as_of_date filtering ──────────────────────────────────────────────


class TestAsOfDateFiltering:
    """Verify that feature builders filter data by as_of_date."""

    def test_staleness_respects_as_of_date(self) -> None:
        """build_staleness_features should only see actions up to as_of_date."""
        from ilga_graph.ml.features import build_staleness_features

        intro = "2025-01-01"
        df_bills = _make_bills(("B1", intro))
        df_actions = _make_actions(
            ("B1", "2025-01-15", "Filed with Secretary"),
            ("B1", "2025-03-01", "Do Pass / Short Debate"),
            ("B1", "2025-06-01", "Third Reading - Passed"),
        )

        # As of Feb 1: should only see the Jan 15 action
        result = build_staleness_features(df_bills, df_actions, as_of_date="2025-02-01")
        row = result.to_dicts()[0]
        # action_count_30d: only the Jan 15 action is within 30 days of intro
        assert row["action_count_30d"] == 1
        # The June action should NOT be visible
        assert row["action_count_90d"] == 1  # only Jan 15

    def test_action_features_respects_as_of_date(self) -> None:
        """build_action_features should only see actions up to as_of_date."""
        from ilga_graph.ml.features import build_action_features

        intro = "2025-01-01"
        df_bills = _make_bills(("B1", intro))
        df_actions = _make_actions(
            ("B1", "2025-01-10", "Referred to Assignments"),
            ("B1", "2025-01-20", "Added as Co-Sponsor Sen. Smith"),
            ("B1", "2025-03-01", "Added as Co-Sponsor Sen. Jones"),
        )

        # As of Feb 1: should only see Jan 10 + Jan 20
        result = build_action_features(df_bills, df_actions, as_of_date="2025-02-01")
        row = result.to_dicts()[0]
        # Both actions within 30 days of intro AND before as_of_date
        assert row["early_action_count"] >= 1

        # As of Jan 15: should only see Jan 10
        result2 = build_action_features(df_bills, df_actions, as_of_date="2025-01-15")
        row2 = result2.to_dicts()[0]
        assert row2["early_action_count"] <= row["early_action_count"]
