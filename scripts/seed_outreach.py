#!/usr/bin/env python3
"""Seed outreach_events from real backlog and (in dev only) mock community data.

- DB path: dev profile → data/ilga_dev.db (sandbox); prod → data/ilga.db (live).
  Set ILGA_PROFILE=dev or =prod (or use .env). Use the SAME profile as the app
  so the app and seed use the same DB; otherwise the heat pill will be empty.
- Real data: seeds backlog by profile — prod → moratyle@gmail.com, dev → funky_mama11@gmail.com
  (rep names → member_id from cache + mocks merged). Only inserts events when the rep name resolves to a
  canonical member id (no slug fallback), so the DB stays consistent with app state.
  Run seed after a scrape so member IDs match the app.
- Mock data: only when ILGA_PROFILE=dev. Seeds advocate1–5@example.com with
  hardcoded numeric member_ids. Also "this week" and fire-pill events. All use
  canonical ids so prod and dev stay consistent.
Run from repo root: python scripts/seed_outreach.py  (or: make seed-outreach)
"""

from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# Backlog: Date, Rep Name, Constituent, Called?, Emailed?, Support Score (1–5), Notes, Contact (from notes).
# Source: user backlog table (Next Action / Task Status columns not stored in outreach_events).
_SEED_ROWS = [
    (
        "1/28/2026",
        "Edgar González",
        False,
        True,
        False,
        3,
        "Spoke with Wenda and understood once I told her why I was calling other senators. Email: office@repedgargonzalez.com",
        "Wenda",
    ),
    (
        "1/27/2026",
        "Diane Blair-Sherlock",
        False,
        True,
        True,
        4,
        "Alicia said her son is is stationed in Japan. Did not know about Kei vehicles",
        "Alicia",
    ),
    (
        "1/26/2026",
        "Ram Villivalam",
        False,
        True,
        True,
        3,
        "Cynthia seemed luke warm on the issue and did not know about Kei vehicles",
        "Cynthia",
    ),
    (
        "1/26/2026",
        "Steve Stadelman",
        False,
        True,
        True,
        3,
        "Howard said he'll contact the SOS to learn more. Email him: howard@senatorstadelman.com",
        "Howard",
    ),
    (
        "1/26/2026",
        "Darby A. Hills",
        False,
        True,
        True,
        3,
        "Deb said to send email to her with more info",
        "Deb",
    ),
    (
        "1/26/2026",
        "Seth Lewis",
        False,
        True,
        True,
        4,
        "Maurine said to speak with my district rep to ask for a legisulative remedy",
        "Maurine",
    ),
    (
        "1/26/2026",
        "Celina Villanueva",
        True,
        True,
        True,
        4,
        "Fatima gave two routes: individual case report and/ or advocacy policy change. Fatima will be leaving for good next week",
        "Fatima",
    ),
    (
        "1/26/2026",
        "Norma Hernandez",
        True,
        True,
        False,  # Emailed? unchecked in spreadsheet
        4,
        "Fernanda did not know what Kei vehicles were and said she'll be in touch. When I told her I was part of a group, she took note and notice",
        "Fernanda",
    ),
    (
        "1/23/2026",
        "Yolonda Morris",
        False,
        True,
        False,
        3,
        "Adriana was very helpful. Said first two weeks of the new year, they take new bills. Now, get a group to keep calling their legisulators. YOU are the ones who need to be educated. The reps just need to look at the facts and move along",
        "Adriana",
    ),
    (
        "1/23/2026",
        "Camille Y. Lilly",
        False,
        True,
        False,
        3,
        "Jaylen answered and said he'll make a note and pass it along",
        "Jaylen",
    ),
    (
        "1/23/2026",
        "Jesus Garcia",
        True,
        True,
        False,
        3,
        "Kyro answered and said to reach out to state reps, Mah and Villanueva",
        "Kyro",
    ),
    (
        "1/23/2026",
        "Jesus Garcia",
        True,
        True,
        False,
        4,
        "Spoke to Samir. While she said she was unfamilar with the issue, she would log the call and encouraged me to contact my D.O offices",
        "Samir",
    ),
    (
        "1/22/2026",
        "Theresa Mah",
        True,
        True,
        True,
        5,
        "Ricky answered and seemed happy and helpful to hear out my issue. He even mentioned that he sees a guy across the street from the office who drives a similar truck",
        "Ricky",
    ),
]

# Dev-only mock: multiple advocates for 60608 + Transportation (heat pill demo).
# (adv_email, zip, list of (member_id, kind, date_str, support_score, notes, contact_name))
_COMMUNITY_ROWS: list[tuple[str, str, list[tuple[str, str, str, int, str, str]]]] = [
    (
        "advocate1@example.com",
        "60608",
        [
            (
                "3291",
                "call",
                "2/10/2026",
                4,
                "Left message with receptionist about kei truck registration",
                "Dana",
            ),
            ("3291", "email", "2/10/2026", 4, "Follow-up email with info packet", "Dana"),
            (
                "3366",
                "call",
                "2/11/2026",
                3,
                "Spoke briefly, was told to send details by email",
                "Marcus",
            ),
            (
                "3318",
                "call",
                "2/12/2026",
                3,
                "Called about transportation policy, staff said to email info",
                "Tina",
            ),
        ],
    ),
    (
        "advocate2@example.com",
        "60608",
        [
            ("3291", "call", "2/12/2026", 3, "Reached staffer who was aware of the issue", "Chris"),
            (
                "3318",
                "call",
                "2/13/2026",
                4,
                "Good conversation about kei trucks on rural roads",
                "Jerome",
            ),
            ("3318", "email", "2/13/2026", 4, "Sent fact sheet and safety data", "Jerome"),
        ],
    ),
    (
        "advocate3@example.com",
        "60616",
        [
            ("3291", "email", "2/14/2026", 3, "Sent constituent letter via website form", ""),
            (
                "3366",
                "call",
                "2/14/2026",
                4,
                "Staffer very interested, asked for more info",
                "Priya",
            ),
            ("3366", "email", "2/14/2026", 4, "Emailed one-pager on kei vehicle benefits", "Priya"),
            ("3318", "call", "2/15/2026", 3, "Brief call, left voicemail", ""),
        ],
    ),
    (
        "advocate4@example.com",
        "60607",
        [
            (
                "3318",
                "call",
                "2/15/2026",
                5,
                "Enthusiastic staffer, said senator is open to hearing more",
                "Val",
            ),
            ("3318", "email", "2/15/2026", 5, "Sent detailed policy brief", "Val"),
            ("3291", "call", "2/16/2026", 4, "Good call, staffer took detailed notes", "Dana"),
        ],
    ),
    (
        "advocate5@example.com",
        "60608",
        [
            ("3366", "call", "2/16/2026", 3, "Quick call, asked to send email instead", "Marcus"),
            ("3318", "email", "2/17/2026", 4, "Sent comparison of state kei vehicle policies", ""),
        ],
    ),
]


def _normalize_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    # NFD and drop combining chars for accent-insensitive match
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s


def _parse_date(date_str: str) -> datetime:
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_str.strip())
    if not m:
        return datetime.now(timezone.utc)
    month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    # Use afternoon (14:00) so times are stable and readable
    return datetime(year, month, day, 14, 0, 0, tzinfo=timezone.utc)


def _load_members_name_to_id() -> tuple[dict[str, str], dict[str, str]]:
    """Load name->member_id from all available members.json and merge.
    Tries CACHE_DIR, then prod cache (cache/), then MOCK_DEV_DIR so numeric IDs
    from full cache are used when available (matches app state when it loads full data).
    """
    from ilga_graph.config import CACHE_DIR, MOCK_DEV_DIR

    _cache_base = CACHE_DIR.parent  # e.g. cache/ when CACHE_DIR is cache/dev
    by_name: dict[str, str] = {}
    by_norm: dict[str, str] = {}
    for base in (CACHE_DIR, _cache_base, MOCK_DEV_DIR):
        path = base / "members.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for m in raw:
            name = (m.get("name") or "").strip()
            mid = (m.get("id") or "").strip()
            if name and mid:
                by_name[name] = mid
                by_norm[_normalize_name(name)] = mid
    return by_name, by_norm


def _resolve_member_id(
    rep_name: str, by_name: dict[str, str], by_norm: dict[str, str]
) -> str | None:
    """Resolve rep name to canonical member id from members.json. Returns None if no match."""
    rep_name = (rep_name or "").strip()
    if rep_name in by_name:
        return by_name[rep_name]
    norm = _normalize_name(rep_name)
    if norm in by_norm:
        return by_norm[norm]
    return None


async def _main() -> None:
    from sqlalchemy import select

    from ilga_graph import config as cfg
    from ilga_graph.db import DB_PATH, async_session_factory, init_db
    from ilga_graph.db_models import OutreachEvent, User

    by_name, by_norm = _load_members_name_to_id()
    if not by_name:
        print(
            "WARNING: No members.json found in cache or mocks. Backlog rows will be skipped "
            "(only dev mock events with hardcoded member_ids will be inserted)."
        )
    else:
        print(f"Resolved {len(by_name)} member names from cache/mocks.")

    await init_db()
    print(f"Database: {DB_PATH} (ILGA_PROFILE={cfg.PROFILE})")

    email = "moratyle@gmail.com" if cfg.PROFILE == "prod" else "funky_mama11@gmail.com"
    async with async_session_factory() as session:
        r = await session.execute(select(User).where(User.email == email))
        user = r.scalar_one_or_none()
        if not user:
            user = User(email=email)
            session.add(user)
            await session.flush()
            print(f"Created user: {user.email} (id={user.id})")
        else:
            print(f"Using existing user: {user.email} (id={user.id})")

        # Replace backlog for this user so re-runs don't duplicate.
        from sqlalchemy import delete

        del_result = await session.execute(
            delete(OutreachEvent).where(OutreachEvent.user_id == user.id)
        )
        removed = del_result.rowcount
        if removed:
            print(f"Removed {removed} existing outreach events for {email} (re-seeding).")
        await session.flush()

        created = 0
        skipped = 0
        for (
            date_str,
            rep_name,
            constituent,
            called,
            emailed,
            support_score,
            notes,
            contact_name,
        ) in _SEED_ROWS:
            member_id = _resolve_member_id(rep_name, by_name, by_norm)
            if member_id is None:
                if called or emailed:
                    skipped += 1
                    print(f"  Skip (no member match): {rep_name!r}")
                continue
            dt = _parse_date(date_str)
            if called:
                session.add(
                    OutreachEvent(
                        user_id=user.id,
                        user_email=user.email,
                        member_id=member_id,
                        kind="call",
                        zip_code=None,
                        outcome=None,
                        notes=notes,
                        contact_name=contact_name or None,
                        support_score=support_score,
                        constituent=constituent,
                        created_at=dt,
                    )
                )
                created += 1
            if emailed:
                session.add(
                    OutreachEvent(
                        user_id=user.id,
                        user_email=user.email,
                        member_id=member_id,
                        kind="email",
                        zip_code=None,
                        outcome=None,
                        notes=notes,
                        contact_name=contact_name or None,
                        support_score=support_score,
                        constituent=constituent,
                        created_at=dt,
                    )
                )
                created += 1
        await session.commit()
        print(
            f"Inserted {created} outreach events for {email}."
            + (f" Skipped {skipped} rows (no member match)." if skipped else "")
        )

        # Dev sandbox only: seed mock community advocates (heat pill demo)
        from ilga_graph import config as cfg

        if cfg.PROFILE == "dev":
            community_created = 0
            for adv_email, adv_zip, events in _COMMUNITY_ROWS:
                r = await session.execute(select(User).where(User.email == adv_email))
                adv_user = r.scalar_one_or_none()
                if not adv_user:
                    adv_user = User(email=adv_email)
                    session.add(adv_user)
                    await session.flush()
                    print(f"  Created community user: {adv_email} (id={adv_user.id})")
                for member_id, kind, date_str, support_score, notes, contact_name in events:
                    dt = _parse_date(date_str)
                    session.add(
                        OutreachEvent(
                            user_id=adv_user.id,
                            user_email=adv_email,
                            member_id=member_id,
                            kind=kind,
                            zip_code=adv_zip,
                            outcome=None,
                            notes=notes,
                            contact_name=contact_name or None,
                            support_score=support_score,
                            constituent=True,
                            created_at=dt,
                        )
                    )
                    community_created += 1
            await session.commit()
            print(
                f"Inserted {community_created} dev mock outreach events across {len(_COMMUNITY_ROWS)} advocates."
            )

            # Dev-only: "this week" events with relative dates so the landing ticker
            # always shows a realistic count (calls_this_week >= 7-day window).
            # (adv_email, member_id, kind, days_ago, support_score, notes, contact_name)
            _THIS_WEEK_ROWS: list[tuple[str, str, str, int, int, str, str]] = [
                (
                    "advocate1@example.com",
                    "3291",
                    "call",
                    6,
                    4,
                    "Mock: left message about kei trucks",
                    "Staff",
                ),
                (
                    "advocate2@example.com",
                    "3318",
                    "call",
                    5,
                    3,
                    "Mock: brief call, asked to email",
                    "",
                ),
                (
                    "advocate3@example.com",
                    "3366",
                    "call",
                    4,
                    4,
                    "Mock: staffer interested",
                    "Priya",
                ),
                (
                    "advocate4@example.com",
                    "3291",
                    "call",
                    3,
                    2,
                    "Mock: skeptical, said to send facts",
                    "Receptionist",
                ),
                (
                    "advocate5@example.com",
                    "3318",
                    "call",
                    2,
                    5,
                    "Mock: champion, will bring up in caucus",
                    "Val",
                ),
                ("advocate1@example.com", "3366", "call", 1, 3, "Mock: neutral, took note", ""),
                (
                    "advocate2@example.com",
                    "3291",
                    "call",
                    0,
                    1,
                    "Mock: strongly opposed to change",
                    "Staff",
                ),
                (
                    "advocate3@example.com",
                    "3318",
                    "call",
                    0,
                    4,
                    "Mock: interested, asked for one-pager",
                    "",
                ),
                ("advocate4@example.com", "3291", "email", 1, 4, "Mock: follow-up after call", ""),
                (
                    "advocate5@example.com",
                    "3366",
                    "call",
                    2,
                    3,
                    "Mock: quick call, will review",
                    "",
                ),
            ]
            # Extra members so more result cards show the fire pill (Power Broker).
            _FIRE_PILL_EXTRA: list[tuple[str, str, int, int, str]] = [
                ("advocate1@example.com", "3292", 3, 4, "Mock: heat pill demo"),
                ("advocate2@example.com", "3292", 2, 3, "Mock: heat pill demo"),
                ("advocate3@example.com", "3312", 4, 4, "Mock: heat pill demo"),
                ("advocate4@example.com", "3279", 1, 3, "Mock: heat pill demo"),
            ]
            now = datetime.now(timezone.utc)
            dev_advocate_emails = [row[0] for row in _COMMUNITY_ROWS]
            r = await session.execute(select(User).where(User.email.in_(dev_advocate_emails)))
            adv_users = {u.email: u for u in r.scalars().all()}
            this_week_created = 0
            for (
                adv_email,
                member_id,
                kind,
                days_ago,
                support_score,
                notes,
                contact_name,
            ) in _THIS_WEEK_ROWS:
                u = adv_users.get(adv_email)
                if not u:
                    continue
                session.add(
                    OutreachEvent(
                        user_id=u.id,
                        user_email=adv_email,
                        member_id=member_id,
                        kind=kind,
                        zip_code="60608",
                        outcome=None,
                        notes=notes,
                        contact_name=contact_name or None,
                        support_score=support_score,
                        constituent=True,
                        created_at=now - timedelta(days=days_ago),
                    )
                )
                this_week_created += 1
            for adv_email, member_id, days_ago, support_score, notes in _FIRE_PILL_EXTRA:
                u = adv_users.get(adv_email)
                if not u:
                    continue
                session.add(
                    OutreachEvent(
                        user_id=u.id,
                        user_email=adv_email,
                        member_id=member_id,
                        kind="call",
                        zip_code="60608",
                        outcome=None,
                        notes=notes,
                        contact_name=None,
                        support_score=support_score,
                        constituent=True,
                        created_at=now - timedelta(days=days_ago),
                    )
                )
                this_week_created += 1
            await session.commit()
            print(
                f"Inserted {this_week_created} dev 'this week' + fire-pill events (relative dates, poll score 1–5)."
            )


if __name__ == "__main__":
    asyncio.run(_main())
