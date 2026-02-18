"""Startup banner table and timing log. Used by main lifespan."""

import logging
from datetime import datetime
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class _Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"


def format_startup_table(
    elapsed_total: float,
    elapsed_load: float,
    elapsed_analytics: float,
    elapsed_seating: float,
    elapsed_export: float,
    elapsed_committee: float,
    elapsed_votes: float,
    elapsed_voting_records: float,
    elapsed_slips: float,
    elapsed_zip: float,
    elapsed_graph: float,
    elapsed_ml: float,
    elapsed_influence: float,
    member_count: int,
    committee_count: int,
    bill_count: int,
    exported_bill_count: int,
    member_committee_role_count: int,
    member_vote_record_count: int,
    category_bill_set_count: int,
    vote_event_count: int,
    slip_count: int,
    bills_with_votes: int,
    bills_with_slips: int,
    zcta_count: int,
    graph_node_count: int,
    graph_edge_count: int,
    ml_prediction_count: int,
    ml_coalition_count: int,
    ml_anomaly_count: int,
    pivotality_count: int,
    sponsor_pull_count: int,
    influence_count: int,
    load_only: bool,
    dev_mode: bool,
    seed_mode: bool,
) -> str:
    """Format a chronological ETL startup table with phase/time/detail."""
    c = _Colors

    def row(phase: str, label: str, sec: float, detail: str) -> str:
        return (
            f"{c.CYAN}{phase:<10}{c.RESET} "
            f"{c.WHITE}{label:<32}{c.RESET}"
            f"{c.BRIGHT_GREEN}{sec:>8.2f}s{c.RESET}  "
            f"{c.WHITE}{detail}{c.RESET}"
        )

    mode_bits = [
        f"load_only={load_only}",
        f"dev_mode={dev_mode}",
        f"seed_mode={seed_mode}",
    ]
    mode_line = ", ".join(mode_bits)

    lines = [
        "",
        f"{c.BOLD}{c.CYAN}{'=' * 100}{c.RESET}",
        f"{c.BOLD}{c.BRIGHT_CYAN}🚀 Application Startup Complete (chronological ETL view){c.RESET}",
        f"{c.DIM}Mode: {mode_line}{c.RESET}",
        f"{c.BOLD}{c.CYAN}{'=' * 100}{c.RESET}",
        "",
        f"{c.BOLD}{'Phase':<10} {'Step':<32} {'Time':>8}  {'Details'}{c.RESET}",
        f"{c.GRAY}{'-' * 100}{c.RESET}",
    ]

    load_detail = f"{member_count} members, {committee_count} committees, {bill_count} bills"
    if load_only:
        load_detail += f" {c.DIM}(cache-only startup){c.RESET}"
    elif seed_mode and elapsed_load < 0.5:
        load_detail += f" {c.DIM}(seed fallback){c.RESET}"
    else:
        load_detail += f" {c.DIM}(cache/scrape){c.RESET}"
    lines.append(row("Extract", "1) Load core data", elapsed_load, load_detail))

    lines.append(
        row(
            "Transform",
            "2) Compute analytics",
            elapsed_analytics,
            f"{member_count} scorecards + Moneyball profiles",
        )
    )
    lines.append(
        row(
            "Transform",
            "3) Seating enrichment",
            elapsed_seating,
            "Senate seat blocks + seatmate affinity",
        )
    )
    export_detail = f"{exported_bill_count} bills exported ({bill_count} in memory)"
    lines.append(row("Load", "4) Export vault artifacts", elapsed_export, export_detail))
    committee_detail = (
        f"{committee_count} committee stats, {member_committee_role_count} members with roles"
    )
    lines.append(row("Transform", "5) Committee indexes", elapsed_committee, committee_detail))
    vote_detail = f"{vote_event_count} vote events"
    if bills_with_votes > 0:
        vote_detail += f" ({bills_with_votes} bills)"
    if elapsed_votes < 0.1 and vote_event_count > 0:
        vote_detail += f" {c.DIM}(cached){c.RESET}"
    lines.append(row("Transform", "6) Vote event index + normalize", elapsed_votes, vote_detail))
    voting_records_detail = (
        f"{member_vote_record_count} members, {category_bill_set_count} category bill sets"
    )
    lines.append(
        row(
            "Transform",
            "7) Member voting records",
            elapsed_voting_records,
            voting_records_detail,
        )
    )
    slip_detail = f"{slip_count} slips"
    if bills_with_slips > 0:
        slip_detail += f" ({bills_with_slips} bills)"
    if elapsed_slips < 0.1 and slip_count > 0:
        slip_detail += f" {c.DIM}(cached){c.RESET}"
    lines.append(row("Transform", "8) Witness slip index", elapsed_slips, slip_detail))
    lines.append(
        row(
            "Reference",
            "9) ZIP district crosswalk",
            elapsed_zip,
            f"{zcta_count} ZCTAs → IL Senate/House districts",
        )
    )
    graph_detail = f"{graph_node_count} nodes, {graph_edge_count} edges"
    lines.append(row("Transform", "10) Co-sponsorship graph", elapsed_graph, graph_detail))
    if ml_prediction_count > 0:
        ml_detail = (
            f"{ml_prediction_count} predictions, "
            f"{ml_coalition_count} coalitions, "
            f"{ml_anomaly_count} anomalies"
        )
    else:
        ml_detail = f"{c.DIM}not available (run make ml-run){c.RESET}"
    lines.append(row("Load", "11) ML intelligence", elapsed_ml, ml_detail))
    inf_detail = (
        f"{pivotality_count} pivotality, "
        f"{sponsor_pull_count} sponsor pull, "
        f"{influence_count} influence profiles"
    )
    lines.append(row("Transform", "12) Influence engine", elapsed_influence, inf_detail))

    lines.extend(
        [
            f"{c.GRAY}{'-' * 100}{c.RESET}",
            f"{c.BOLD}{'Total':<43}{c.BRIGHT_CYAN}{elapsed_total:>8.2f}s{c.RESET}  "
            f"{c.DIM}{mode_line}{c.RESET}",
            f"{c.BOLD}{c.CYAN}{'=' * 100}{c.RESET}",
            "",
        ]
    )
    return "\n".join(lines)


def log_startup_timing(
    total_s: float,
    load_s: float,
    analytics_s: float,
    seating_s: float,
    export_s: float,
    votes_s: float,
    slips_s: float,
    zip_s: float,
    graph_s: float,
    ml_s: float,
    influence_s: float,
    member_count: int,
    bill_count: int,
    vote_count: int,
    slip_count: int,
    zcta_count: int,
    dev_mode: bool,
    seed_mode: bool,
) -> None:
    """Append startup timing to .startup_timings.csv for historical tracking."""
    log_file = Path(".startup_timings.csv")
    is_new = not log_file.exists()
    with open(log_file, "a", encoding="utf-8") as f:
        if is_new:
            f.write(
                "timestamp,total_s,load_s,analytics_s,seating_s,export_s,votes_s,slips_s,zip_s,"
                "graph_s,ml_s,influence_s,"
                "members,bills,votes,slips,zctas,dev_mode,seed_mode\n"
            )
        f.write(
            f"{datetime.now().isoformat()},{total_s:.2f},{load_s:.2f},{analytics_s:.2f},"
            f"{seating_s:.2f},{export_s:.2f},{votes_s:.2f},{slips_s:.2f},{zip_s:.2f},"
            f"{graph_s:.2f},{ml_s:.2f},{influence_s:.2f},"
            f"{member_count},{bill_count},{vote_count},{slip_count},{zcta_count},"
            f"{dev_mode},{seed_mode}\n"
        )
    LOGGER.debug("Startup timing logged to %s", log_file)
