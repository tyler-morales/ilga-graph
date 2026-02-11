from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from .analytics import MemberScorecard, compute_scorecard
from .models import Bill, Committee, CommitteeMemberRole, Member, Office
from .moneyball import MoneyballProfile, MoneyballReport

LOGGER = logging.getLogger(__name__)

_RE_CODE = re.compile(r"^code:\s*(\S+)", re.MULTILINE)
_RE_PARENT_CODE = re.compile(r"^parent_code:\s*(\S+)", re.MULTILINE)
_RE_SLUG = re.compile(r"[^a-z0-9]+")
_RE_SAFE_FILENAME = re.compile(r"[^\w \-&.',]")


class ObsidianExporter:
    def __init__(
        self,
        vault_root: Path | str = Path("ILGA_Graph_Vault"),
        committees: Iterable[Committee] | None = None,
        committee_rosters: dict[str, list[CommitteeMemberRole]] | None = None,
        committee_bills: dict[str, list[str]] | None = None,
        member_export_limit: int | None = None,
        committee_export_limit: int | None = None,
        bill_export_limit: int | None = None,
    ) -> None:
        self.vault_root = Path(vault_root)
        self.committees = list(committees) if committees else []
        self.committee_lookup = {c.code: c for c in self.committees}
        self.committee_rosters = committee_rosters or {}
        self.committee_bills = committee_bills or {}
        self.member_export_limit = member_export_limit
        self.committee_export_limit = committee_export_limit
        self.bill_export_limit = bill_export_limit

    def export(
        self,
        members: Iterable[Member],
        *,
        scorecards: dict[str, MemberScorecard] | None = None,
        moneyball: MoneyballReport | None = None,
        all_bills: Iterable[Bill] | None = None,
    ) -> None:
        members = list(members)
        member_lookup = {m.id: m for m in members}
        members_path = self.vault_root / "Members"
        members_path.mkdir(parents=True, exist_ok=True)
        if not self.committee_lookup:
            self._load_committees_from_vault(self.vault_root / "Committees")

        # Build a leg_id → Bill lookup so member pages can resolve IDs to bill numbers
        bills_by_leg_id: dict[str, Bill] = {}
        if all_bills is not None:
            for b in all_bills:
                bills_by_leg_id[b.leg_id] = b

        # ── Clean up orphaned root-level bill files ──
        _BILL_PREFIX_RE = re.compile(r"^[HS][BRJ]", re.IGNORECASE)
        _SAFE_ROOT_FILES = {"Moneyball Report.md", "Scorecard Guide.md"}
        for path in self.vault_root.glob("*.md"):
            if path.name in _SAFE_ROOT_FILES:
                continue
            if _BILL_PREFIX_RE.match(path.stem):
                LOGGER.info("Removing orphaned root-level bill file: %s", path.name)
                path.unlink()

        # ── Members ───────────────────────────────────────────────────
        members_to_write = list(members)
        if self.member_export_limit is not None:
            members_to_write.sort(
                key=lambda m: (
                    scorecards[m.id].effectiveness_score
                    if scorecards and m.id in scorecards
                    else compute_scorecard(m).effectiveness_score
                ),
                reverse=True,
            )
            members_to_write = members_to_write[: self.member_export_limit]
        else:
            current_files = {f"{self._safe_filename(m.name)}.md" for m in members}
            for path in members_path.glob("*.md"):
                if path.name not in current_files:
                    path.unlink()

        for member in members_to_write:
            sc = scorecards.get(member.id) if scorecards else None
            mb = moneyball.profiles.get(member.id) if moneyball else None
            file_path = members_path / f"{self._safe_filename(member.name)}.md"
            file_path.write_text(
                self._render_member(
                    member,
                    scorecard=sc,
                    moneyball_profile=mb,
                    bills_lookup=bills_by_leg_id,
                ),
                encoding="utf-8",
            )

        LOGGER.info("Exported %d member files.", len(members_to_write))

        # ── Committees ────────────────────────────────────────────────
        if self.committees:
            committees_path = self.vault_root / "Committees"
            committees_path.mkdir(parents=True, exist_ok=True)

            committees_to_write = list(self.committees)
            if self.committee_export_limit is not None:
                committees_to_write = committees_to_write[: self.committee_export_limit]
            else:
                current_committee_files = {
                    f"{self._safe_filename(c.name)}.md" for c in self.committees
                }
                for path in committees_path.glob("*.md"):
                    if path.name not in current_committee_files:
                        path.unlink()

            members_by_committee = self._members_by_committee(members)
            children_by_parent = self._children_by_parent_committee()
            for committee in committees_to_write:
                file_path = committees_path / f"{self._safe_filename(committee.name)}.md"
                content = self._render_committee(
                    committee,
                    members_by_committee,
                    children_by_parent,
                    member_lookup,
                    self.committee_bills.get(committee.code, []),
                )
                file_path.write_text(content, encoding="utf-8")

            LOGGER.info("Exported %d committee files.", len(committees_to_write))

        # ── Bill set: built exclusively from all_bills (separation of concerns) ──
        bills_path = self.vault_root / "Bills"
        bills_path.mkdir(parents=True, exist_ok=True)

        unique_bills: dict[str, Bill] = {}
        bill_cosponsors: dict[str, list[str]] = {}
        if all_bills is not None:
            for bill in all_bills:
                bn = bill.bill_number
                if bn not in unique_bills:
                    unique_bills[bn] = bill
                    bill_cosponsors[bn] = []

        # Build cosponsor names by iterating members' co_sponsor_bill_ids
        for member in members:
            for lid in member.co_sponsor_bill_ids:
                bill = bills_by_leg_id.get(lid)
                if bill and bill.bill_number in bill_cosponsors:
                    bill_cosponsors[bill.bill_number].append(member.name)

        # Determine which bills to write to disk.
        bills_to_write = list(unique_bills.items())
        if self.bill_export_limit is not None:

            def _bill_sort_key(item: tuple[str, Bill]) -> datetime:
                try:
                    return datetime.strptime(item[1].last_action_date, "%m/%d/%Y")
                except (ValueError, TypeError):
                    return datetime.min

            bills_to_write.sort(key=_bill_sort_key, reverse=True)
            bills_to_write = bills_to_write[: self.bill_export_limit]
        else:
            current_bill_files = {f"{bn}.md" for bn in unique_bills}
            for path in bills_path.glob("*.md"):
                if path.name not in current_bill_files:
                    path.unlink()

        for bill_number, bill in bills_to_write:
            file_path = bills_path / f"{bill_number}.md"
            file_path.write_text(
                self._render_bill(bill, bill_cosponsors.get(bill_number, [])),
                encoding="utf-8",
            )

        LOGGER.info(
            "Exported %d bill files (of %d total in memory).",
            len(bills_to_write),
            len(unique_bills),
        )

        index_path = members_path / "ILGA_Member_Index.md"
        index_path.write_text(self._render_index(members), encoding="utf-8")

        # ── Moneyball Report ──────────────────────────────────────────────
        if moneyball is not None:
            report_path = self.vault_root / "Moneyball Report.md"
            report_path.write_text(
                self._render_moneyball_report(moneyball, member_lookup),
                encoding="utf-8",
            )
            LOGGER.info("Exported Moneyball Report.")

        # ── Obsidian Bases database views ─────────────────────────────────
        self._write_base_files()

    # ── rendering ────────────────────────────────────────────────────────

    def _resolve_bill_links(self, bill_ids: list[str], bills_lookup: dict[str, Bill]) -> str:
        """Resolve a list of leg_ids to [[bill_number]] wikilinks."""
        links = []
        for lid in bill_ids:
            bill = bills_lookup.get(lid)
            if bill:
                links.append(f"- [[{bill.bill_number}]]")
        return "\n".join(links) if links else "- None"

    def _render_member(
        self,
        member: Member,
        scorecard: MemberScorecard | None = None,
        moneyball_profile: MoneyballProfile | None = None,
        bills_lookup: dict[str, Bill] | None = None,
    ) -> str:
        if scorecard is None:
            scorecard = compute_scorecard(member)
        if bills_lookup is None:
            bills_lookup = {}
        tags = self._build_tags(member)
        career_start_year = ""
        if member.career_ranges:
            career_start_year = str(min(cr.start_year for cr in member.career_ranges))

        # Moneyball frontmatter fields
        mb_fields = ""
        if moneyball_profile is not None:
            mb = moneyball_profile
            mb_fields = (
                f"moneyball_score: {mb.moneyball_score}\n"
                f"pipeline_depth: {mb.pipeline_depth_avg}\n"
                f"network_centrality: {mb.network_centrality}\n"
                f"unique_collaborators: {mb.unique_collaborators}\n"
                f"is_leadership: {str(mb.is_leadership).lower()}\n"
                f"rank_overall: {mb.rank_overall}\n"
                f"rank_chamber: {mb.rank_chamber}\n"
                f"rank_non_leadership: {mb.rank_non_leadership}\n"
            )

        frontmatter = (
            "---\n"
            f"chamber: {member.chamber}\n"
            f"party: {member.party}\n"
            f"role: {member.role}\n"
            f"career_timeline: {member.career_timeline_text}\n"
            f"career_start_year: {career_start_year}\n"
            f"district: {member.district}\n"
            f"member_url: {member.member_url}\n"
            f"bills_introduced: {scorecard.law_heat_score}\n"
            f"laws_passed: {scorecard.law_passed_count}\n"
            f"law_success_rate: {scorecard.law_success_rate}\n"
            f"magnet_score: {scorecard.magnet_score}\n"
            f"bridge_score: {scorecard.bridge_score}\n"
            f"resolutions_filed: {scorecard.resolutions_count}\n"
            f"resolutions_passed: {scorecard.resolutions_passed_count}\n"
            f"resolution_pass_rate: {scorecard.resolution_pass_rate}\n"
            f"total_primary_bills: {scorecard.primary_bill_count}\n"
            f"total_passed: {scorecard.passed_count}\n"
            f"overall_pass_rate: {scorecard.success_rate}\n"
            f"{mb_fields}"
            f"tags: [{', '.join(tags)}]\n"
            "---\n\n"
        )

        committees_links = (
            "\n".join(f"- {self._committee_link(c)}" for c in member.committees) or "- None"
        )

        contact_blocks = [self._render_office_block(o) for o in member.offices]
        if member.email:
            contact_blocks.append(f"### Email\n- {member.email}")
        contact_section = (
            "\n\n".join(contact_blocks) if contact_blocks else "No contact info found."
        )

        # ── Legislation sections (IDs only — separation of concerns) ──
        primary_links = self._resolve_bill_links(member.sponsored_bill_ids, bills_lookup)
        cosponsorship_links = self._resolve_bill_links(member.co_sponsor_bill_ids, bills_lookup)

        scorecard_section = self._render_scorecard(scorecard)
        moneyball_section = (
            self._render_moneyball_section(moneyball_profile) if moneyball_profile else ""
        )

        body = (
            f"# {member.name}\n\n"
            "## ILGA Page\n"
            f"{member.member_url}\n\n"
            "## Chamber\n"
            f"{member.chamber}\n\n"
            "## Role\n"
            f"{member.role or 'None'}\n\n"
            "## Career Timeline\n"
            f"{self._render_career_ranges(member)}\n\n"
            "## Scorecard\n"
            "See [[Scorecard Guide]] for how these metrics are defined.\n\n"
            f"{scorecard_section}\n\n"
            f"{moneyball_section}"
            "## Tags\n"
            f"{self._render_tag_lines(tags)}\n\n"
            "## Committees\n"
            f"{committees_links}\n\n"
            "## \U0001f4dc Primary Legislation\n"
            f"{primary_links}\n\n"
            "## \U0001f58a\ufe0f Co-Sponsorships\n"
            f"{cosponsorship_links}\n\n"
            "## Contact\n"
            f"{contact_section}\n\n"
            f"{self._render_associated_section(member)}\n\n"
            "## Biography\n"
            f"{member.bio_text or 'None'}\n"
        )

        return frontmatter + body

    def _render_bill(self, bill: Bill, cosponsors: list[str] | None = None) -> str:
        chamber_tag = "senate" if bill.chamber == "S" else "house"
        iso_date = ""
        try:
            parsed = datetime.strptime(bill.last_action_date, "%m/%d/%Y")
            iso_date = parsed.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            iso_date = ""
        frontmatter = (
            "---\n"
            f'leg_id: "{bill.leg_id}"\n'
            f"bill_number: {bill.bill_number}\n"
            f"chamber: {bill.chamber}\n"
            f'status: "{bill.last_action}"\n'
            f'last_action_date: "{bill.last_action_date}"\n'
            f"last_action_date_iso: {iso_date}\n"
            f"tags: [type/bill, chamber/{chamber_tag}]\n"
            "---\n\n"
        )

        cosponsor_lines = (
            "\n".join(f"- [[{name}]]" for name in sorted(cosponsors)) if cosponsors else "- None"
        )

        # ── Actions section from action_history ──
        if bill.action_history:
            action_lines = "\n".join(
                f"| {a.date} | {a.chamber} | {a.action} |" for a in bill.action_history
            )
            actions_section = (
                f"\n## Actions\n| Date | Chamber | Action |\n| --- | --- | --- |\n{action_lines}\n"
            )
        else:
            actions_section = "\n## Actions\nNo actions recorded.\n"

        body = (
            f"# {bill.bill_number}\n\n"
            "## Description\n"
            f"{bill.description}\n\n"
            "## Primary Sponsor\n"
            f"[[{bill.primary_sponsor}]]\n\n"
            "## Co-Sponsors\n"
            f"{cosponsor_lines}\n\n"
            "## Status\n"
            f"{bill.last_action} ({bill.last_action_date})\n"
            f"{actions_section}"
        )

        return frontmatter + body

    def _safe_filename(self, name: str) -> str:
        result = _RE_SAFE_FILENAME.sub("", name).strip()
        return result if result else "unknown"

    def _render_office_block(self, office: Office) -> str:
        blocks: list[str] = []
        address_lines = [line for line in office.address.splitlines() if line.strip()]
        if address_lines:
            blocks.append("Address:\n" + "\n".join(f"- {line}" for line in address_lines))
        if office.phone:
            blocks.append("Phone:\n- " + office.phone)
        if office.fax:
            blocks.append("Fax:\n- " + office.fax)
        return f"### {office.name}\n" + "\n".join(blocks)

    def _render_career_ranges(self, member: Member) -> str:
        if member.career_ranges:
            rendered: list[str] = []
            for cr in member.career_ranges:
                end_text = "Present" if cr.end_year is None else str(cr.end_year)
                entry = f"{cr.start_year} - {end_text}"
                if cr.chamber:
                    entry = f"{entry} ({cr.chamber})"
                rendered.append(f"- {entry}")
            return "\n".join(rendered)
        return member.career_timeline_text or "None"

    def _render_scorecard(self, sc: MemberScorecard) -> str:
        law_pct = f"{sc.law_success_rate * 100:.1f}%"
        magnet_str = f"{sc.magnet_score:.1f}"
        bridge_pct = f"{sc.bridge_score * 100:.1f}%"
        res_pass_pct = f"{sc.resolution_pass_rate * 100:.1f}%"
        overall_pct = f"{sc.success_rate * 100:.1f}%"

        sections = (
            "### Lawmaking (HB/SB)\n"
            "| Metric | Value | Formula |\n"
            "| --- | --- | --- |\n"
            f"| Bills Introduced | {sc.law_heat_score} | Count of primary HB/SB |\n"
            f"| Passed | {sc.law_passed_count} | HB/SB that became law |\n"
            f"| Success Rate | {law_pct} | Passed ÷ Bills Introduced |\n"
            f"| Avg Co-Sponsors (Magnet) | {magnet_str} | Co-sponsors ÷ Introduced |\n"
            f"| Cross-Party (Bridge) | {bridge_pct} | Cross-party co-sponsor ÷ Introduced |\n\n"
            "### Resolutions (HR/SR/HJR/SJR)\n"
            "| Metric | Value | Formula |\n"
            "| --- | --- | --- |\n"
            f"| Resolutions Filed | {sc.resolutions_count} | Count of primary HR/SR/HJR/SJR |\n"
            f"| Passed | {sc.resolutions_passed_count} | Resolutions adopted |\n"
            f"| Pass Rate | {res_pass_pct} | Passed ÷ Resolutions Filed |\n\n"
            "### Overall\n"
            "| Metric | Value | Formula |\n"
            "| --- | --- | --- |\n"
            f"| Total Primary Bills | {sc.primary_bill_count} | Introduced + Resolutions Filed |\n"
            f"| Total Passed | {sc.passed_count} | Laws Passed + Resolutions Passed |\n"
            f"| Overall Pass Rate | {overall_pct} | Total Passed ÷ Total Primary Bills |"
        )

        # ── Context interpretation badges ──
        badges: list[str] = []
        if sc.magnet_score > 10:
            badges.append("\U0001f525 **Coalition Builder** (High co-sponsorship avg)")
        if sc.bridge_score > 0.2:
            badges.append("\U0001f91d **Bipartisan Bridge** (>20% cross-party support)")
        if sc.resolutions_count > sc.law_heat_score:
            badges.append("\U0001f4e2 **Ceremonial Focus**")

        if badges:
            sections += "\n\n" + "\n".join(badges)

        return sections

    # ── tags ─────────────────────────────────────────────────────────────

    def _build_tags(self, member: Member) -> list[str]:
        tags = [
            "type/member",
            f"chamber/{member.chamber.lower()}",
            f"party/{member.party.lower()}",
        ]
        tags.extend(self._committee_tags(member.committees))
        return list(dict.fromkeys(tags))

    def _committee_tags(self, committees: Iterable[str]) -> list[str]:
        tags: list[str] = []
        for committee_code in committees:
            parent_code = self._committee_parent_code(committee_code)
            committee_name = self._committee_name(committee_code)
            if parent_code:
                parent_name = self._committee_name(parent_code)
                parent_slug = self._slug(parent_name)
                child_slug = self._slug(committee_name)
                tags.append(f"committee/{parent_slug}")
                tags.append(f"subcommittee/{parent_slug}/{child_slug}")
            else:
                tags.append(f"committee/{self._slug(committee_name)}")
        return tags

    def _committee_parent_code(self, committee_code: str) -> str | None:
        committee = self.committee_lookup.get(committee_code)
        if committee and committee.parent_code:
            return committee.parent_code
        if "-" in committee_code:
            return committee_code.split("-", 1)[0]
        return None

    def _committee_link(self, committee_code: str) -> str:
        committee = self.committee_lookup.get(committee_code)
        return f"[[{committee.name}]]" if committee else f"[[{committee_code}]]"

    def _committee_name(self, committee_code: str) -> str:
        committee = self.committee_lookup.get(committee_code)
        return committee.name if committee else committee_code

    def _render_tag_lines(self, tags: list[str]) -> str:
        return "\n".join(f"#{tag}" for tag in tags) if tags else "#tags/none"

    # ── associated members ───────────────────────────────────────────────

    def _render_associated_section(self, member: Member) -> str:
        """Render associated members section with chamber-specific label."""
        is_senate = member.chamber.lower() == "senate"
        if is_senate:
            header = "## Associated Representatives"
        else:
            header = "## Associated Senator"

        content = self._render_associated_links(member.associated_members)
        return f"{header}\n{content}"

    def _render_associated_links(self, associated_members: str | None) -> str:
        if not associated_members:
            return "None"
        names = [n.strip() for n in associated_members.split(",") if n.strip()]
        return "\n".join(f"- [[{name}]]" for name in names) if names else "None"

    # ── index ────────────────────────────────────────────────────────────

    def _render_index(self, members: Iterable[Member]) -> str:
        members_list = sorted(members, key=lambda m: m.name)
        links = "\n".join(f"- [[{m.name}]] ([ILGA]({m.member_url}))" for m in members_list)
        tag_pool: set[str] = set()
        for member in members_list:
            tag_pool.update(self._build_tags(member))
        tag_lines = "\n".join(f"#{tag}" for tag in sorted(tag_pool))
        return (
            "---\n"
            "tags: [index, ilga, members]\n"
            "---\n\n"
            "# ILGA Member Index\n\n"
            "## Guides\n"
            "- [[Scorecard Guide]] — How to read the scorecard and each metric.\n\n"
            "## Members\n"
            f"{links}\n\n"
            "## Tags\n"
            f"{tag_lines}\n"
        )

    # ── committee rendering ──────────────────────────────────────────────

    def _members_by_committee(self, members: Iterable[Member]) -> dict[str, list[Member]]:
        result: dict[str, list[Member]] = {}
        for member in members:
            for code in member.committees:
                result.setdefault(code, []).append(member)
        return result

    def _children_by_parent_committee(self) -> dict[str, list[Committee]]:
        result: dict[str, list[Committee]] = {}
        for committee in self.committees:
            if committee.parent_code:
                result.setdefault(committee.parent_code, []).append(committee)
        return result

    def _render_committee(
        self,
        committee: Committee,
        members_by_committee: dict[str, list[Member]],
        children_by_parent: dict[str, list[Committee]],
        member_lookup: dict[str, Member],
        committee_bills: list[str] | None = None,
    ) -> str:
        tag_values = list(
            dict.fromkeys(
                [
                    "type/committee",
                    "committee" if committee.parent_code is None else "subcommittee",
                    *self._committee_tags([committee.code]),
                ]
            )
        )
        frontmatter = (
            "---\n"
            f"code: {committee.code}\n"
            f"parent_code: {committee.parent_code or ''}\n"
            f"tags: [{', '.join(tag_values)}]\n"
            "---\n\n"
        )

        parent_link = "None"
        if committee.parent_code:
            parent = self.committee_lookup.get(committee.parent_code)
            parent_link = f"[[{parent.name}]]" if parent else committee.parent_code

        roster_entries = self.committee_rosters.get(committee.code, [])
        if roster_entries:
            members_section = "\n".join(
                f"- [[{self._resolve_member_name(entry, member_lookup)}]] ({entry.role})"
                for entry in roster_entries
            )
        else:
            member_links = members_by_committee.get(committee.code, [])
            members_section = (
                "\n".join(f"- [[{m.name}]]" for m in member_links) if member_links else "- None"
            )

        subcommittees = children_by_parent.get(committee.code, [])
        subcommittee_section = (
            "\n".join(f"- [[{child.name}]]" for child in subcommittees)
            if subcommittees
            else "- None"
        )

        bills_section = (
            "\n".join(f"- [[{bn}]]" for bn in committee_bills) if committee_bills else "- None"
        )

        body = (
            f"# {committee.name}\n\n"
            "## Code\n"
            f"{committee.code}\n\n"
            "## Parent Committee\n"
            f"{parent_link}\n\n"
            "## Subcommittees\n"
            f"{subcommittee_section}\n\n"
            "## Members\n"
            f"{members_section}\n\n"
            "## Bills\n"
            f"{bills_section}\n"
        )

        return frontmatter + body

    def _resolve_member_name(
        self, entry: CommitteeMemberRole, member_lookup: dict[str, Member]
    ) -> str:
        if entry.member_id and entry.member_id in member_lookup:
            return member_lookup[entry.member_id].name
        return entry.member_name or "Unknown"

    # ── vault loading ────────────────────────────────────────────────────

    def _load_committees_from_vault(self, committees_path: Path) -> None:
        if not committees_path.exists():
            return
        committees: list[Committee] = []
        for path in committees_path.glob("*.md"):
            content = path.read_text(encoding="utf-8")
            code_match = _RE_CODE.search(content)
            if not code_match:
                continue
            parent_match = _RE_PARENT_CODE.search(content)
            parent_code = parent_match.group(1).strip() if parent_match else None
            if parent_code in ("", "None", "null"):
                parent_code = None
            committees.append(
                Committee(
                    code=code_match.group(1).strip(),
                    name=path.stem,
                    parent_code=parent_code,
                )
            )
        self.committees = committees
        self.committee_lookup = {c.code: c for c in committees}

    # ── Moneyball rendering ─────────────────────────────────────────────

    def _render_moneyball_section(self, mb: MoneyballProfile) -> str:
        """Render the Moneyball analytics section for a member note."""
        depth_bar = self._depth_progress_bar(mb.pipeline_depth_avg, 6)
        badge_line = ", ".join(mb.badges) if mb.badges else "None yet"

        section = (
            "## Moneyball Analytics\n"
            "See [[Moneyball Report]] for full rankings and methodology.\n\n"
            "| Metric | Value |\n"
            "| --- | --- |\n"
            f"| Moneyball Score | **{mb.moneyball_score}** / 100 |\n"
            f"| Rank (Overall) | #{mb.rank_overall} |\n"
            f"| Rank ({mb.chamber}) | #{mb.rank_chamber} |\n"
        )
        if not mb.is_leadership:
            section += f"| Rank (Non-Leadership) | #{mb.rank_non_leadership} |\n"
        section += (
            f"| Leadership | {'Yes' if mb.is_leadership else 'No'} |\n"
            f"| Pipeline Depth | {mb.pipeline_depth_avg:.1f} / 6.0 {depth_bar} |\n"
            f"| Network Centrality | {mb.network_centrality:.2%} |\n"
            f"| Unique Collaborators | {mb.unique_collaborators} |\n"
            f"| Badges | {badge_line} |\n\n"
        )
        return section

    def _depth_progress_bar(self, value: float, max_val: float, width: int = 6) -> str:
        """Render a text progress bar like [####..]."""
        filled = round(value / max_val * width) if max_val > 0 else 0
        return "[" + "#" * filled + "." * (width - filled) + "]"

    def _render_moneyball_report(
        self,
        report: MoneyballReport,
        member_lookup: dict[str, Member],
    ) -> str:
        """Render the full Moneyball Report as an Obsidian note."""
        w = report.weights_used
        frontmatter = "---\ntags: [ilga, moneyball, analytics, rankings]\n---\n\n"

        # Header
        body = "# The Moneyball Report\n\n"
        body += (
            '> *"Can we identify the most effective legislator in the House '
            'who is not in leadership?"*\n\n'
        )

        # ── MVP callout ──
        if report.mvp_house_non_leadership:
            mvp = report.profiles[report.mvp_house_non_leadership]
            body += (
                "## MVP: Most Effective Non-Leadership House Member\n\n"
                f"**[[{mvp.member_name}]]** — District {mvp.district} ({mvp.party})\n\n"
                f"| Metric | Value |\n"
                f"| --- | --- |\n"
                f"| Moneyball Score | **{mvp.moneyball_score}** / 100 |\n"
                f"| Laws Filed (HB/SB) | {mvp.laws_filed} |\n"
                f"| Laws Passed | {mvp.laws_passed} |\n"
                f"| Effectiveness Rate | {mvp.effectiveness_rate:.1%} |\n"
                f"| Avg Co-Sponsors (Magnet) | {mvp.magnet_score:.1f} |\n"
                f"| Cross-Party Rate (Bridge) | {mvp.bridge_score:.1%} |\n"
                f"| Pipeline Depth | {mvp.pipeline_depth_avg:.1f} / 6.0 |\n"
                f"| Network Centrality | {mvp.network_centrality:.2%} |\n"
                f"| Unique Collaborators | {mvp.unique_collaborators} |\n"
                f"| Badges | {', '.join(mvp.badges) or 'None'} |\n\n"
            )

        if report.mvp_senate_non_leadership:
            mvp_s = report.profiles[report.mvp_senate_non_leadership]
            body += (
                "## MVP: Most Effective Non-Leadership Senate Member\n\n"
                f"**[[{mvp_s.member_name}]]** — District {mvp_s.district} ({mvp_s.party})\n\n"
                f"| Metric | Value |\n"
                f"| --- | --- |\n"
                f"| Moneyball Score | **{mvp_s.moneyball_score}** / 100 |\n"
                f"| Laws Filed (HB/SB) | {mvp_s.laws_filed} |\n"
                f"| Laws Passed | {mvp_s.laws_passed} |\n"
                f"| Effectiveness Rate | {mvp_s.effectiveness_rate:.1%} |\n"
                f"| Pipeline Depth | {mvp_s.pipeline_depth_avg:.1f} / 6.0 |\n"
                f"| Badges | {', '.join(mvp_s.badges) or 'None'} |\n\n"
            )

        # ── Methodology ──
        body += (
            "---\n\n"
            "## Methodology\n\n"
            "The Moneyball Score is a weighted composite of five normalized metrics:\n\n"
            "| Component | Weight | What It Measures |\n"
            "| --- | --- | --- |\n"
            f"| Effectiveness | {w.effectiveness:.0%} | Laws Passed / Laws Filed (HB/SB) |\n"
            f"| Pipeline Depth | {w.pipeline:.0%} | How far bills progress (0=filed, 6=signed) |\n"
            f"| Magnet Score | {w.magnet:.0%} | Avg co-sponsors per law (normalized) |\n"
            f"| Bridge Score | {w.bridge:.0%} | Cross-party co-sponsorship rate |\n"
            f"| Network Centrality | {w.centrality:.0%} | Co-sponsorship graph degree |\n\n"
            "**No-Fluff Filter**: Only substantive legislation (HB/SB) counts for "
            "effectiveness and pipeline metrics. Resolutions (HR/SR/HJR/SJR) are "
            "tracked separately.\n\n"
            "**Leadership Filter**: Members with formal leadership titles (Speaker, "
            'Leader, Whip, etc.) are ranked separately so we can surface "hidden gems."\n\n'
        )

        # ── Full Rankings ──
        body += "---\n\n## House Rankings (Non-Leadership)\n\n"
        body += self._render_ranking_table(report, report.rankings_house_non_leadership)
        body += "\n\n## Senate Rankings (Non-Leadership)\n\n"
        body += self._render_ranking_table(report, report.rankings_senate_non_leadership)
        body += "\n\n## House Rankings (All)\n\n"
        body += self._render_ranking_table(report, report.rankings_house)
        body += "\n\n## Senate Rankings (All)\n\n"
        body += self._render_ranking_table(report, report.rankings_senate)

        return frontmatter + body

    def _render_ranking_table(
        self,
        report: MoneyballReport,
        member_ids: list[str],
        limit: int = 50,
    ) -> str:
        """Render a ranking table for a list of member IDs."""
        header = (
            "| # | Member | Score | Laws | Eff% | Magnet | Bridge | Depth | Centrality | Badges |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
        rows: list[str] = []
        for rank, mid in enumerate(member_ids[:limit], 1):
            p = report.profiles[mid]
            badges = ", ".join(p.badges[:3]) if p.badges else "-"
            ldp = " *" if p.is_leadership else ""
            rows.append(
                f"| {rank} | [[{p.member_name}]]{ldp} | {p.moneyball_score} | "
                f"{p.laws_passed}/{p.laws_filed} | {p.effectiveness_rate:.0%} | "
                f"{p.magnet_score:.1f} | {p.bridge_score:.0%} | "
                f"{p.pipeline_depth_avg:.1f} | {p.network_centrality:.2%} | {badges} |"
            )
        return header + "\n".join(rows) if rows else "No members in this category."

    # ── Obsidian Bases generation ───────────────────────────────────────

    def _write_base_files(self) -> None:
        """Generate Obsidian Bases (.base) database view files."""
        bills_base = (
            'filters: file.inFolder("Bills")\n'
            "views:\n"
            "  - type: table\n"
            '    name: "Bills by Date"\n'
            "    groupBy:\n"
            "      property: last_action_date_iso\n"
            "      direction: DESC\n"
            "    order:\n"
            "      - last_action_date_iso\n"
            "      - bill_number\n"
            "      - status\n"
            "      - chamber\n"
            "  - type: table\n"
            '    name: "Recent Bills"\n'
            f"    filters: 'note.last_action_date_iso >= date(\"{datetime.now().year}-01-01\")'\n"
            "    groupBy:\n"
            "      property: last_action_date_iso\n"
            "      direction: DESC\n"
            "    order:\n"
            "      - last_action_date_iso\n"
            "      - bill_number\n"
            "      - status\n"
            "      - chamber\n"
        )
        (self.vault_root / "Bills by Date.base").write_text(bills_base, encoding="utf-8")

        members_base = (
            "filters:\n"
            "  and:\n"
            '    - file.inFolder("Members")\n'
            "    - 'file.name != \"ILGA_Member_Index\"'\n"
            "views:\n"
            "  - type: table\n"
            '    name: "Members by Career Start"\n'
            "    groupBy:\n"
            "      property: career_start_year\n"
            "      direction: ASC\n"
            "    order:\n"
            "      - career_start_year\n"
            "      - chamber\n"
            "      - party\n"
            "      - role\n"
            "      - file.name\n"
        )
        (self.vault_root / "Members by Career.base").write_text(members_base, encoding="utf-8")

        heat_base = (
            "filters:\n"
            "  and:\n"
            '    - file.inFolder("Members")\n'
            "    - 'file.name != \"ILGA_Member_Index\"'\n"
            "views:\n"
            "  - type: table\n"
            '    name: "Legislative Heat"\n'
            "    order:\n"
            "      - property: total_primary_bills\n"
            "        direction: DESC\n"
            "    columns:\n"
            "      - file.name\n"
            "      - total_primary_bills\n"
            "      - total_passed\n"
            "      - overall_pass_rate\n"
            "      - bills_introduced\n"
            "      - laws_passed\n"
            "      - law_success_rate\n"
            "      - magnet_score\n"
            "      - bridge_score\n"
            "      - party\n"
            "      - district\n"
            "  - type: table\n"
            '    name: "Most Effective"\n'
            "    order:\n"
            "      - property: total_passed\n"
            "        direction: DESC\n"
            "    columns:\n"
            "      - file.name\n"
            "      - total_passed\n"
            "      - law_success_rate\n"
            "      - total_primary_bills\n"
            "      - magnet_score\n"
            "      - bridge_score\n"
            "      - bills_introduced\n"
            "      - resolutions_filed\n"
            "      - party\n"
            "      - district\n"
        )
        (self.vault_root / "Members by Heat Score.base").write_text(heat_base, encoding="utf-8")

        moneyball_base = (
            "filters:\n"
            "  and:\n"
            '    - file.inFolder("Members")\n'
            "    - 'file.name != \"ILGA_Member_Index\"'\n"
            "    - 'note.moneyball_score > 0'\n"
            "views:\n"
            "  - type: table\n"
            '    name: "Moneyball Leaderboard"\n'
            "    order:\n"
            "      - property: moneyball_score\n"
            "        direction: DESC\n"
            "    columns:\n"
            "      - file.name\n"
            "      - moneyball_score\n"
            "      - rank_chamber\n"
            "      - laws_passed\n"
            "      - law_success_rate\n"
            "      - pipeline_depth\n"
            "      - magnet_score\n"
            "      - bridge_score\n"
            "      - network_centrality\n"
            "      - unique_collaborators\n"
            "      - is_leadership\n"
            "      - party\n"
            "      - chamber\n"
            "  - type: table\n"
            '    name: "Hidden Gems (Non-Leadership)"\n'
            "    filters: 'note.is_leadership == false'\n"
            "    order:\n"
            "      - property: moneyball_score\n"
            "        direction: DESC\n"
            "    columns:\n"
            "      - file.name\n"
            "      - moneyball_score\n"
            "      - rank_non_leadership\n"
            "      - laws_passed\n"
            "      - law_success_rate\n"
            "      - pipeline_depth\n"
            "      - magnet_score\n"
            "      - bridge_score\n"
            "      - network_centrality\n"
            "      - party\n"
            "      - chamber\n"
        )
        (self.vault_root / "Moneyball Leaderboard.base").write_text(
            moneyball_base, encoding="utf-8"
        )

    def _slug(self, value: str) -> str:
        return _RE_SLUG.sub("-", value.lower()).strip("-") or "unknown"
