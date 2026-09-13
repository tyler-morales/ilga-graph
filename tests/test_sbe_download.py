"""SBE download fallback (Range 403 → full GET) and ingest-job wiring."""

from __future__ import annotations

import email.message
import io
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from ilga_graph.campaign_finance import download as sbe_download

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.headers = {"Content-Length": str(len(body))}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _http_error(url: str, code: int, reason: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url,
        code,
        reason,
        email.message.Message(),
        io.BytesIO(b""),
    )


def _patch_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sbe_download.time, "sleep", lambda _seconds: None)


@pytest.mark.parametrize("status_code", [403, 416])
def test_download_chunked_falls_back_to_full_get_when_range_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    url = "https://downloads.elections.il.gov/Candidates.txt"
    body = b"ID\tName\n1\tLightford\n"
    dest = tmp_path / "Candidates.txt"
    _patch_sleep(monkeypatch)

    def fake_urlopen(req: urllib.request.Request, timeout: object = None) -> _FakeResponse:
        if req.has_header("Range"):
            reason = "Forbidden" if status_code == 403 else "Range Not Satisfiable"
            raise _http_error(url, status_code, reason)
        return _FakeResponse(body)

    monkeypatch.setattr(sbe_download.urllib.request, "urlopen", fake_urlopen)

    result = sbe_download.download_chunked(url, dest, size=len(body))

    assert result == dest
    assert dest.read_bytes() == body


def test_download_chunked_does_not_full_get_receipts_tail_on_range_403(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = "https://downloads.elections.il.gov/Receipts.txt"
    dest = tmp_path / "Receipts.txt"
    _patch_sleep(monkeypatch)

    def fake_urlopen(req: urllib.request.Request, timeout: object = None) -> _FakeResponse:
        if req.has_header("Range"):
            raise _http_error(url, 403, "Forbidden")
        return _FakeResponse(b"should-not-write-1gb")

    monkeypatch.setattr(sbe_download.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(sbe_download.RangeBlockedError, match="Cloudflare") as excinfo:
        sbe_download.download_chunked(url, dest, size=25_000_000, start_at=1_000_000)

    assert "Receipts.txt" in str(excinfo.value)
    assert not dest.exists() or dest.stat().st_size == 0


def test_download_range_sends_browser_accept_and_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}
    _patch_sleep(monkeypatch)

    def fake_urlopen(req: urllib.request.Request, timeout: object = None) -> _FakeResponse:
        captured.update({k.lower(): v for k, v in req.header_items()})
        return _FakeResponse(b"ok")

    monkeypatch.setattr(sbe_download.urllib.request, "urlopen", fake_urlopen)

    payload = sbe_download.download_range(
        "https://downloads.elections.il.gov/Candidates.txt", 0, 10
    )

    assert payload == b"ok"
    assert "mozilla" in captured["user-agent"].lower()
    assert "text/plain" in captured["accept"]
    assert captured["range"] == "bytes=0-10"


def test_ci_ingest_job_downloads_on_runner_and_parses_from_dir_on_pi() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "if: github.event_name == 'workflow_dispatch'" in text
    assert "timeout-minutes: 60" in text
    assert "download_sbe_files" in text
    assert "pip install -e ." in text
    assert "name: Download SBE files on the runner" in text
    assert "rsync" in text
    ssh = text.split("name: Ingest SBE money via SSH", 1)[1]
    assert "FROM_DIR=cache/sbe" in ssh
    assert "download_sbe_files" not in ssh
    assert "elections.il.gov" not in ssh
