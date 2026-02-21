"""Scene registry for dev component playground. Add entries to expose components in isolation."""

from __future__ import annotations

from . import advocacy_helpers as ah

# Type for a scene: id, label, template, context dict, optional trigger key for JS.
_SCENES: list[dict] = []


def _drawer_call_context() -> dict:
    """Mock context for drawer call partial (no real member)."""
    return {
        "legislator_name": "Jane Smith",
        "zip_code": "60601",
        "is_constituent": True,
        "phone": "(217) 555-0100",
        "member_id": "playground-mock",
        "photo_url": "",
        "member_public_email": "",
        "target_type": "NON_COMMITTEE",
        **ah.legislator_drawer_context(None),
    }


def _drawer_email_context() -> dict:
    """Mock context for drawer email partial (no real member)."""
    legislator_name = "Jane Smith"
    zip_code = "60601"
    subject_constituent = ah.build_email_subject_line(zip_code, variant="constituent")
    subject_general = ah.build_email_subject_line(zip_code, variant="general")
    body = ah.build_email_first_body(
        legislator_name,
        zip_code,
        chamber="House",
        district="5",
        target_type="NON_COMMITTEE",
    )
    body_followup = ah.build_after_call_email_body(
        "",
        legislator_name,
        zip_code,
        chamber="House",
        district="5",
        target_type="NON_COMMITTEE",
        call_date="",
    )
    return {
        "drawer_view": "email_first",
        "legislator_name": legislator_name,
        "legislator_display_name": ah.get_legislator_display_name(legislator_name, "House", "5"),
        "recipient_email": "",
        "contact_name": "",
        "has_public_email": False,
        "subject": subject_constituent,
        "subject_constituent": subject_constituent,
        "subject_general": subject_general,
        "body": body,
        "body_followup": body_followup,
        "body_first": body,
        "show_call_nudge": True,
        "show_go_to_call": True,
        "zip_code": zip_code,
        "is_constituent": True,
        "party_abbr": "D",
    }


def get_scenes() -> list[dict]:
    """Return the list of playground scenes. Context may be built lazily for templates."""
    if _SCENES:
        return _SCENES
    _SCENES.extend(
        [
            {
                "id": "truck",
                "label": "Truck animation",
                "template": "dev_playground/_scene_truck.html",
                "context": {},
                "trigger": "truck",
            },
            {
                "id": "drawer-call",
                "label": "Drawer (call view)",
                "template": "_dev_playground_drawer_call.html",
                "context": _drawer_call_context,
                "trigger": None,
            },
            {
                "id": "drawer-email",
                "label": "Drawer (email view)",
                "template": "_dev_playground_drawer_email.html",
                "context": _drawer_email_context,
                "trigger": None,
            },
        ]
    )
    return _SCENES


def get_scene(scene_id: str) -> dict | None:
    """Return the scene dict for scene_id, or None if not found."""
    scenes = get_scenes()
    for s in scenes:
        if s.get("id") == scene_id:
            return s
    return None


def get_scene_context(scene: dict, request):  # noqa: ANN001
    """Resolve context for a scene: call context callable or return dict; request is always set."""
    ctx = scene.get("context") or {}
    if callable(ctx):
        ctx = ctx()
    ctx = dict(ctx) if isinstance(ctx, dict) else {}
    ctx["request"] = request
    return ctx
