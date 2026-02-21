"""Tests for photo_url query parameter validation in the advocacy drawer.

Verifies that validate_photo_url_for_drawer only allows safe URLs (relative paths
or same-origin ilga.gov) and rejects protocol-relative, javascript:, data:, and
other origins that could be used for XSS or loading external resources in <img src>.
"""

from __future__ import annotations

import pytest

from ilga_graph.security import validate_photo_url_for_drawer


class TestValidatePhotoUrlForDrawer:
    """Unit tests for validate_photo_url_for_drawer."""

    @pytest.mark.parametrize(
        "url",
        [
            None,
            "",
            "   ",
        ],
    )
    def test_empty_or_none_returns_none(self, url: str | None) -> None:
        assert validate_photo_url_for_drawer(url) is None

    @pytest.mark.parametrize(
        "url",
        [
            "/legislation/some/path",
            "/Members/photo.jpg",
            "/",
        ],
    )
    def test_safe_relative_paths_allowed(self, url: str) -> None:
        assert validate_photo_url_for_drawer(url) == url

    def test_relative_path_with_dotdot_rejected(self) -> None:
        assert validate_photo_url_for_drawer("/legislation/../etc/passwd") is None
        assert validate_photo_url_for_drawer("/..") is None

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.ilga.gov/legislation/104/SB/photo.jpg",
            "https://www.ilga.gov/",
            "https://www.ilga.gov",
        ],
    )
    def test_ilga_gov_https_allowed(self, url: str) -> None:
        assert validate_photo_url_for_drawer(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "//evil.com/image.png",
            "//www.ilga.gov/legislation/photo.jpg",
            "javascript:alert(1)",
            "javascript:",
            "data:text/html,<script>alert(1)</script>",
            "data:image/png;base64,xxx",
            "vbscript:msgbox(1)",
            "https://evil.com/image.png",
            "http://evil.com/image.png",
            "https://www.ilga.gov.evil.com/x",
        ],
    )
    def test_dangerous_or_external_rejected(self, url: str) -> None:
        assert validate_photo_url_for_drawer(url) is None, f"Expected {url!r} to be rejected"

    def test_overlong_url_rejected(self) -> None:
        long_path = "/" + "a" * 2048
        assert validate_photo_url_for_drawer(long_path) is None
