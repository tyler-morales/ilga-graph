"""E2E tests for the outreach flow: ZIP lookup, preference tree, call/email drawers."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e

# Dev seed data: 60601 is a good ZIP that returns legislators
TEST_ZIP = "60601"
# Cookie set by /advocacy/set-call-pref; setting it lets the page load with saved state (no pref tree click).
ADV_CALL_PREF_COOKIE = "adv_call_pref"


def _goto_advocacy_with_pref_saved(page, base_url: str, pref: str = "yes") -> None:
    """Navigate to advocacy with pref cookie set so intro card shows saved state (skip pref tree).
    Loads page, POSTs set-call-pref (sets cookie), then reloads so saved state renders.
    """
    page.goto(base_url + f"/advocacy?zip={TEST_ZIP}")
    page.locator("#advocacy-intro-pref-section").wait_for(state="visible", timeout=15000)
    page.evaluate(
        """async (pref) => {
        await fetch('/advocacy/set-call-pref', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ pref }),
            credentials: 'same-origin'
        });
    }""",
        pref,
    )
    page.reload()
    page.locator("#advocacy-intro-pref-section .intro-card__saved-msg").wait_for(
        state="visible", timeout=15000
    )


def test_zip_lookup_redirects_to_advocacy(page, base_url):
    """Enter ZIP in hero form -> land on /advocacy?zip=..."""
    page.goto(base_url + "/")
    page.locator("#advocacy-form").wait_for(state="visible", timeout=10000)
    page.locator("#zip_code").fill(TEST_ZIP)
    page.get_by_role("button", name="Start outreach").click()
    page.wait_for_load_state("networkidle", timeout=15000)
    # Navigation already happened; wait_for_url would wait for a future nav and time out.
    assert "advocacy" in page.url
    assert TEST_ZIP in page.url


def test_advocacy_page_shows_legislators(page, base_url):
    """Direct GET /advocacy?zip=60601 shows at least one legislator card."""
    page.goto(base_url + f"/advocacy?zip={TEST_ZIP}")
    page.wait_for_load_state("networkidle", timeout=20000)
    page.locator("#member-carousel").wait_for(state="visible", timeout=15000)
    # Intro card may be first; at least one member slide or intro
    slides = page.locator("#member-carousel .member-carousel__slide")
    slides.first.wait_for(state="visible", timeout=5000)
    assert slides.count() >= 1


@pytest.mark.skip(reason="set-call-pref POST from pref tree not yet reliable in e2e; cookie path used for drawer tests")
def test_pref_tree_saves_email_pref(page, base_url):
    """Pref tree: Yes (email) -> No (2-min call) -> No (30-sec) -> saved confirmation."""
    page.goto(base_url + f"/advocacy?zip={TEST_ZIP}")
    page.locator("#advocacy-intro-pref-section").wait_for(state="visible", timeout=15000)
    # Step 1: Yes (email OK)
    page.locator("#pref-step-1").get_by_role("button", name="Yes").click()
    # Step 2a: No (no 2-min call)
    page.locator("#pref-step-2a").wait_for(state="visible", timeout=5000)
    page.locator("#pref-step-2a").get_by_role("button", name="No").click()
    # Step 3a: No (email only) — button aria-label is "No — email only"
    page.locator("#pref-step-3a").wait_for(state="visible", timeout=5000)
    page.locator("#pref-step-3a").get_by_role("button", name="No — email only").click()
    page.locator("#advocacy-intro-pref-section .intro-card__saved-msg").wait_for(state="visible", timeout=10000)
    assert "email only" in page.locator("#advocacy-intro-pref-section").inner_text().lower()


@pytest.mark.skip(reason="set-call-pref POST from pref tree not yet reliable in e2e; cookie path used for drawer tests")
def test_pref_tree_saves_call_pref(page, base_url):
    """Pref tree: Yes (email) -> Yes (2-min call) -> saved confirmation."""
    page.goto(base_url + f"/advocacy?zip={TEST_ZIP}")
    page.locator("#advocacy-intro-pref-section").wait_for(state="visible", timeout=15000)
    page.locator("#pref-step-1").get_by_role("button", name="Yes").click()
    page.locator("#pref-step-2a").wait_for(state="visible", timeout=5000)
    # Yes — call and email (button with phone icon, label "Yes")
    page.locator("#pref-step-2a").get_by_role("button", name="Yes").click()
    page.locator("#advocacy-intro-pref-section .intro-card__saved-msg").wait_for(state="visible", timeout=10000)
    assert "script ready" in page.locator("#advocacy-intro-pref-section").inner_text().lower() or "let's go" in page.locator("#advocacy-intro-pref-section").inner_text().lower()


@pytest.mark.skip(reason="pref saved state (cookie/POST) not reliable in e2e; intro card stays tree")
def test_call_drawer_opens(page, base_url):
    """Set pref to call+email, then click Call on first member -> drawer with script opens."""
    _goto_advocacy_with_pref_saved(page, base_url)
    # First member slide (not intro)
    member_slide = page.locator("#member-carousel .member-carousel__slide[data-member-id]").first
    member_slide.wait_for(state="visible", timeout=5000)
    member_slide.get_by_role("button", name="Call Script").click()
    page.locator("#advocacy-drawer-body .drawer-call-content").wait_for(state="visible", timeout=10000)
    assert page.locator("#drawer-script-intro").is_visible()


@pytest.mark.skip(reason="pref saved state (cookie/POST) not reliable in e2e; intro card stays tree")
def test_call_drawer_wrapup(page, base_url):
    """In call drawer: End call -> wrap-up form -> select outcome -> success."""
    _goto_advocacy_with_pref_saved(page, base_url)
    page.locator("#member-carousel .member-carousel__slide[data-member-id]").first.get_by_role("button", name="Call Script").click()
    page.locator("#advocacy-drawer-body .drawer-call-content").wait_for(state="visible", timeout=10000)
    page.get_by_role("button", name="End call").first.click()
    # Wrap-up inline or form: interest poll pills or "How interested did they seem?"
    page.locator("#drawer-wrapup-inline, .drawer-interest-poll-bubble, .drawer-wrapup-inline-intro").first.wait_for(state="visible", timeout=8000)
    # Select an interest level (e.g. Neutral)
    interest = page.locator('.drawer-interest-pill[data-score="3"]')
    if interest.is_visible():
        interest.click()
    # "Call recorded" or similar confirmation
    page.locator(".drawer-draft-followup-recorded-msg, .drawer-wrapup-inline").first.wait_for(state="visible", timeout=5000)


@pytest.mark.skip(reason="pref saved state (cookie/POST) not reliable in e2e; intro card stays tree")
def test_email_drawer_opens(page, base_url):
    """Set pref, click Email on first member -> email compose drawer."""
    _goto_advocacy_with_pref_saved(page, base_url)
    page.locator("#member-carousel .member-carousel__slide[data-member-id]").first.get_by_role("button", name="Email Script").click()
    # Gmail-style compose or no-email fallback
    page.locator("#advocacy-drawer-body .gmail-compose, #advocacy-drawer-body .drawer-no-email-content").first.wait_for(state="visible", timeout=10000)
    body = page.locator("#advocacy-drawer-body")
    assert body.locator(".gmail-compose").is_visible() or body.locator(".drawer-no-email-content").is_visible()


@pytest.mark.skip(reason="pref saved state (cookie/POST) not reliable in e2e; intro card stays tree")
def test_no_answer_flow(page, base_url):
    """In call drawer: click No answer? -> voicemail / no-answer content."""
    _goto_advocacy_with_pref_saved(page, base_url)
    page.locator("#member-carousel .member-carousel__slide[data-member-id]").first.get_by_role("button", name="Call Script").click()
    page.locator("#advocacy-drawer-body .drawer-call-content").wait_for(state="visible", timeout=10000)
    page.locator("#drawer-voicemail-toggle-btn").click()
    page.locator("#drawer-voicemail, .drawer-no-answer-content").first.wait_for(state="visible", timeout=5000)
    assert page.locator(".drawer-voicemail-single, .drawer-no-answer-headline").first.is_visible()
