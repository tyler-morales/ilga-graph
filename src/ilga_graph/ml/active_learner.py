"""Interactive human-in-the-loop entity resolution trainer.

Presents ambiguous vote-name-to-member matches to the user via a Rich CLI.
User confirms/rejects matches, and the system learns and propagates corrections.
Persists all confirmed mappings to ``processed/entity_gold.json``.

Usage::

    python scripts/ml_resolve.py           # Interactive session
    python scripts/ml_resolve.py --auto     # Auto-resolve high-confidence only
    python scripts/ml_resolve.py --stats    # Show resolution stats only
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .entity_resolution import (
    MemberCandidate,
    ResolutionReport,
    apply_resolution,
    fuzzy_match_candidates,
    load_gold_mappings,
    resolve_all_names,
    save_gold_mappings,
    save_resolved_casts,
)

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = Path("processed")

console = Console()


# ── Stats display ────────────────────────────────────────────────────────────


def display_stats(report: ResolutionReport) -> None:
    """Show a rich summary table of resolution stats."""
    table = Table(title="Entity Resolution Summary", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Rate", justify="right")

    total = report.total_unique_names
    table.add_row("Unique (name, chamber) pairs", str(total), "")
    table.add_row(
        "Resolved (exact match)",
        str(report.resolved_exact),
        f"{100 * report.resolved_exact / total:.1f}%" if total else "N/A",
    )
    table.add_row(
        "Resolved (fuzzy auto)",
        str(report.resolved_fuzzy),
        f"{100 * report.resolved_fuzzy / total:.1f}%" if total else "N/A",
    )
    table.add_row(
        "Resolved (gold/user)",
        str(report.resolved_gold),
        f"{100 * report.resolved_gold / total:.1f}%" if total else "N/A",
    )
    table.add_row(
        "[bold green]Total resolved[/]",
        f"[bold green]{report.total_resolved}[/]",
        f"[bold green]{100 * report.resolution_rate:.1f}%[/]" if total else "N/A",
    )
    table.add_row(
        "[bold red]Unresolved[/]",
        f"[bold red]{report.unresolved}[/]",
        f"[bold red]{100 * report.unresolved / total:.1f}%[/]" if total else "N/A",
    )

    console.print()
    console.print(table)
    console.print()


# ── Interactive session ──────────────────────────────────────────────────────


def _format_candidate(i: int, c: MemberCandidate) -> str:
    """Format a candidate for display."""
    party_color = "blue" if c.party == "Democrat" else "red" if c.party == "Republican" else "white"
    return (
        f"  [{party_color}][{i}][/{party_color}] "
        f"{c.name} ({c.party[0]}-{c.chamber}"
        f"{f'-{c.district}' if c.district else ''}) "
        f"-- score: {c.score:.0f}"
    )


def interactive_session(
    df_vote_casts: pl.DataFrame,
    df_members: pl.DataFrame,
    *,
    batch_size: int = 50,
) -> ResolutionReport:
    """Run an interactive entity resolution session.

    Presents unresolved names one at a time, sorted by occurrence count
    (most common first -- highest impact). User picks the correct member
    or skips. After each batch, shows updated stats and offers to continue.
    """
    # Initial resolution pass
    report = resolve_all_names(df_vote_casts, df_members)
    display_stats(report)

    unresolved = [r for r in report.results if r.method == "unresolved"]
    if not unresolved:
        console.print("[bold green]All names resolved! Nothing to review.[/]")
        return report

    console.print(
        f"[bold yellow]{len(unresolved)} unresolved names[/] "
        f"(sorted by frequency -- highest impact first)\n"
    )

    # Sort by occurrence count descending
    unresolved.sort(key=lambda r: -r.occurrence_count)

    gold = load_gold_mappings()
    resolved_count = 0
    skipped_count = 0
    rejected_count = 0

    for batch_start in range(0, len(unresolved), batch_size):
        batch = unresolved[batch_start : batch_start + batch_size]

        for i, result in enumerate(batch, start=batch_start + 1):
            # Find fuzzy candidates
            candidates = fuzzy_match_candidates(
                result.raw_name,
                result.chamber,
                df_members,
                top_n=5,
                threshold=30.0,
            )

            # Display
            console.print(
                Panel(
                    f'[bold]"{result.raw_name}"[/] '
                    f"({result.chamber}, {result.occurrence_count} votes)",
                    title=f"[{i}/{len(unresolved)}] Unresolved Name",
                    border_style="yellow",
                )
            )

            if candidates:
                for j, c in enumerate(candidates, 1):
                    console.print(_format_candidate(j, c))
            else:
                console.print("  [dim]No candidates found[/]")

            console.print("\n  [dim][s] Skip  [x] Not a real member  [q] Quit & save[/]")

            # Get user input
            while True:
                choice = console.input("\n  Your choice: ").strip().lower()

                if choice == "q":
                    save_gold_mappings(gold)
                    console.print(
                        f"\n[bold green]Session saved.[/] "
                        f"Resolved: {resolved_count}, "
                        f"Skipped: {skipped_count}, "
                        f"Rejected: {rejected_count}"
                    )
                    # Re-run resolution with updated gold
                    report = resolve_all_names(df_vote_casts, df_members)
                    display_stats(report)
                    return report

                if choice == "s":
                    skipped_count += 1
                    break

                if choice == "x":
                    # Mark as "not a member" -- store with empty member_id
                    gold[(result.raw_name, result.chamber)] = "__NOT_A_MEMBER__"
                    rejected_count += 1
                    break

                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(candidates):
                        selected = candidates[idx]
                        gold[(result.raw_name, result.chamber)] = selected.member_id
                        resolved_count += 1
                        console.print(
                            f"  [green]Confirmed: {result.raw_name} "
                            f"-> {selected.name} ({selected.member_id})[/]"
                        )
                        break
                    else:
                        console.print("  [red]Invalid choice. Try again.[/]")
                except ValueError:
                    console.print("  [red]Enter a number, 's', 'x', or 'q'.[/]")

            console.print()

        # End of batch -- save and show stats
        save_gold_mappings(gold)
        report = resolve_all_names(df_vote_casts, df_members)
        display_stats(report)

        remaining = report.unresolved
        if remaining == 0:
            console.print("[bold green]All names resolved![/]")
            break

        console.print(f"[yellow]{remaining} names still unresolved.[/]")
        cont = console.input("Continue? [Y/n]: ").strip().lower()
        if cont == "n":
            break

    # Final save and resolution
    save_gold_mappings(gold)
    report = resolve_all_names(df_vote_casts, df_members)

    # Apply and save resolved casts
    df_resolved = apply_resolution(df_vote_casts, report)
    save_resolved_casts(df_resolved)

    display_stats(report)
    return report


# ── Auto-resolve mode ────────────────────────────────────────────────────────


def auto_resolve(
    df_vote_casts: pl.DataFrame,
    df_members: pl.DataFrame,
    *,
    fuzzy_threshold: float = 95.0,
) -> ResolutionReport:
    """Run resolution without human input (exact + high-confidence fuzzy only).

    Useful for CI or first-pass resolution.
    """
    report = resolve_all_names(
        df_vote_casts,
        df_members,
        fuzzy_auto_threshold=fuzzy_threshold,
    )

    # Apply and save
    df_resolved = apply_resolution(df_vote_casts, report)
    save_resolved_casts(df_resolved)

    display_stats(report)
    return report
