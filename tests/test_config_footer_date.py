"""Footer last-updated date: env override, git fallback, default."""

import re

import pytest

from ilga_graph.config import (
    FOOTER_LAST_UPDATED,
    FOOTER_LAST_UPDATED_ISO,
    _footer_last_updated_from_git,
)


def test_footer_last_updated_from_git_in_repo_returns_formatted_date() -> None:
    """When run inside the repo, returns (human, iso) with expected format."""
    result = _footer_last_updated_from_git()
    if result is None:
        pytest.skip("not in a git repo or git unavailable")
    human, iso = result
    assert re.match(r"^[A-Za-z]+ \d{1,2}, \d{4}$", human), human
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", iso), iso


def test_footer_last_updated_from_git_when_git_fails_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When git is unavailable or fails, returns None."""
    import subprocess

    def raise_(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", raise_)
    assert _footer_last_updated_from_git() is None


def test_footer_last_updated_iso_format() -> None:
    """Resolved footer ISO date is always YYYY-MM-DD (env, git, or default)."""
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", FOOTER_LAST_UPDATED_ISO), FOOTER_LAST_UPDATED_ISO


def test_footer_last_updated_non_empty() -> None:
    """Resolved human date is non-empty."""
    assert FOOTER_LAST_UPDATED.strip()
