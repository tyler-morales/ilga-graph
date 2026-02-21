"""Tests for ZIP code input validation in the advocacy router.

Verifies that the _ZIP_RE pattern (and the endpoint logic that uses it) correctly
accepts valid 5-digit Illinois ZIP codes and rejects malformed input that could
otherwise be reflected into page content (issue #22).
"""

from __future__ import annotations

import pytest

# Import the compiled pattern directly so we can test it in isolation without
# spinning up the full FastAPI app or requiring the database.
from ilga_graph.routers.advocacy import _ZIP_RE


class TestZipRegex:
    """Unit tests for the _ZIP_RE validation pattern."""

    @pytest.mark.parametrize(
        "zip_code",
        ["60601", "60071", "00000", "99999", "12345"],
    )
    def test_valid_five_digit_zips(self, zip_code: str) -> None:
        assert _ZIP_RE.match(zip_code), f"Expected {zip_code!r} to be valid"

    @pytest.mark.parametrize(
        "bad_input",
        [
            "",
            "6060",  # too short
            "606011",  # too long
            "60071fff",  # letters appended (from issue #22 example)
            "<script>",  # script tag
            "1234x",  # letter in zip
            " 60601",  # leading space
            "60601 ",  # trailing space
            "abc12",  # letters
            "12345\x00",  # null byte
        ],
    )
    def test_invalid_inputs_rejected(self, bad_input: str) -> None:
        assert not _ZIP_RE.match(bad_input), f"Expected {bad_input!r} to be rejected"
