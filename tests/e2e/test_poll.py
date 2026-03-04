"""E2E tests for the kei poll: home page and standalone /poll."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


def test_home_poll_renders(page, base_url):
    """Home page loads and the poll flow container is visible."""
    page.goto(base_url + "/")
    flow = page.locator("#why-you-care-flow")
    flow.wait_for(state="visible", timeout=10000)
    assert flow.is_visible()
    # Poll form or branch content should be present
    assert page.locator(
        "#why-you-care-flow form, #why-you-care-flow .wyc-branch-content"
    ).first.is_visible()


def test_home_poll_step1_yes(page, base_url):
    """Click Yes on step 1 -> step 2 (registered/revoked/denied) appears."""
    page.goto(base_url + "/")
    page.locator("#why-you-care-flow").wait_for(state="visible", timeout=10000)
    page.locator("#home-kei-poll-form").get_by_role("button", name="Yes").click()
    panel = page.locator("#home-kei-poll-panel-have")
    panel.wait_for(state="visible", timeout=5000)
    assert panel.is_visible()
    assert page.get_by_label("Registered", exact=True).is_visible()


def test_home_poll_step1_no(page, base_url):
    """Click No on step 1 -> step 2 (would want / would not) appears."""
    page.goto(base_url + "/")
    page.locator("#why-you-care-flow").wait_for(state="visible", timeout=10000)
    page.locator("#home-kei-poll-form").get_by_role("button", name="No").click()
    panel = page.locator("#home-kei-poll-panel-no")
    panel.wait_for(state="visible", timeout=5000)
    assert panel.is_visible()
    assert page.get_by_text("Yes, I'd want one", exact=False).first.is_visible()


def test_home_poll_submit_anon(page, base_url):
    """Complete poll (Yes -> registered -> impact -> submit) -> success or results visible."""
    page.goto(base_url + "/")
    page.locator("#why-you-care-flow").wait_for(state="visible", timeout=10000)
    page.locator("#home-kei-poll-form").get_by_role("button", name="Yes").click()
    page.locator("#home-kei-poll-panel-have").wait_for(state="visible", timeout=5000)
    page.get_by_label("Registered", exact=True).check()
    # Step 3 impact
    impact_panel = page.locator("#home-kei-poll-panel-impact")
    impact_panel.wait_for(state="visible", timeout=5000)
    page.locator('input[name="kei_impact_radio"]').first.check()
    # ZIP may be required on home in some configs; fill if present
    zip_input = page.locator("#home-kei-poll-zip-inline")
    if zip_input.is_visible():
        zip_input.fill("60601")
    submit = page.locator("#home-kei-poll-form").get_by_role(
        "button", name="Submit and see results"
    )
    submit.wait_for(state="visible", timeout=3000)
    # Button may be enabled by JS after impact selection
    page.wait_for_timeout(300)
    submit.click()
    # HTMX swaps #why-you-care-flow with success/branch content
    page.locator(
        "#why-you-care-flow .kei-status-success, #why-you-care-flow .wyc-branch-content"
    ).first.wait_for(state="visible", timeout=10000)
    assert (
        page.locator("#why-you-care-flow")
        .locator(".kei-status-success, .wyc-branch-content")
        .first.is_visible()
    )


def test_standalone_poll_renders(page, base_url):
    """GET /poll shows the poll form."""
    page.goto(base_url + "/poll")
    form = page.locator("#standalone-kei-poll-form, .kei-poll__form")
    form.wait_for(state="visible", timeout=10000)
    assert form.is_visible()
    assert page.get_by_text("Do you have a kei vehicle?", exact=False).first.is_visible()


def test_standalone_poll_submit(page, base_url):
    """Complete standalone poll -> results or thanks visible."""
    page.goto(base_url + "/poll")
    page.locator("#standalone-kei-poll-form").wait_for(state="visible", timeout=10000)
    page.get_by_role("button", name="Yes").first.click()
    page.locator("#standalone-kei-poll-panel-have").wait_for(state="visible", timeout=5000)
    page.get_by_label("Registered", exact=True).check()
    page.locator("#standalone-kei-poll-panel-impact").wait_for(state="visible", timeout=5000)
    page.locator('input[name="kei_impact_radio"]').first.check()
    # Standalone requires ZIP
    page.locator("#standalone-kei-poll-zip").fill("60601")
    submit = page.locator("#standalone-kei-poll-form").get_by_role(
        "button", name="Submit and see results"
    )
    submit.wait_for(state="visible", timeout=3000)
    page.wait_for_timeout(300)
    submit.click()
    # HTMX swaps innerHTML of #standalone-kei-poll-wrap with _kei_poll_anonymous_success (no full redirect).
    page.locator(
        "#standalone-kei-poll-results-drawer, .kei-poll-drawer, .poll-standalone__results"
    ).first.wait_for(state="visible", timeout=10000)
    assert (
        page.locator(".kei-poll-drawer").first.is_visible()
        or page.locator(".poll-standalone__results").first.is_visible()
    )
