"""Bill-related API routes: SHAP explanation fragment (lazy-loaded by HTMX)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from ..app_state import state
from ..models import Bill

_LOGGER = logging.getLogger(__name__)
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
router = APIRouter()
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


def _enrich_explanation_factors(
    result: dict,
    bill_id: str,
    bill: Bill | None,
    score: Any | None,
) -> None:
    """Add concrete bill/score details to explanation factors (e.g. co-sponsor count and names)."""
    member_by_id = getattr(state, "member_lookup_by_id", None) or {}

    def detail_for(raw_feature: str) -> str | None:
        if raw_feature == "sponsor_count" and bill and bill.sponsor_ids:
            n = max(0, len(bill.sponsor_ids) - 1)
            if n == 0:
                return "0 co-sponsors"
            names = []
            for mid in bill.sponsor_ids[1:][:8]:
                m = member_by_id.get(mid)
                names.append(m.name if m else f"ID {mid}")
            if n > len(names):
                return f"{n} co-sponsors: " + ", ".join(names) + f", and {n - len(names)} more"
            return f"{n} co-sponsors: " + ", ".join(names)

        if raw_feature == "days_since_last_action" and score is not None:
            d = getattr(score, "days_since_action", None)
            if d is not None:
                return f"{int(d)} days since last movement"
            if getattr(score, "last_action_text", None):
                return (score.last_action_text or "")[:50]

        if raw_feature == "days_since_intro" and score is not None:
            intro = getattr(score, "introduction_date", None)
            if intro:
                return f"Introduced {intro}"

        if raw_feature in ("sponsor_party", "sponsor_party_democrat", "sponsor_party_republican"):
            if bill and bill.primary_sponsor and member_by_id:
                for mid in (bill.sponsor_ids or [])[:1]:
                    m = member_by_id.get(mid)
                    if m:
                        return m.party or bill.primary_sponsor
                return bill.primary_sponsor
            return None

        if raw_feature == "sponsor_hist_passage_rate" and bill and bill.sponsor_ids:
            primary_id = bill.sponsor_ids[0] if bill.sponsor_ids else None
            if primary_id and member_by_id.get(primary_id):
                return f"Primary sponsor: {bill.primary_sponsor}"
            return None

        return None

    for factors in (
        result.get("top_positive_factors", []),
        result.get("top_negative_factors", []),
    ):
        for f in factors:
            raw = f.get("raw_feature")
            if raw:
                detail = detail_for(raw)
                if detail:
                    f["detail"] = detail


@router.get("/bills/{bill_id}/explanation")
async def bill_explanation_fragment(request: Request, bill_id: str):
    """Return an HTML fragment with SHAP prediction drivers for a bill.

    Designed to be loaded lazily via hx-get so the main bill page
    renders instantly and SHAP computation happens in the background.
    """
    ml = state.ml
    if not ml or not ml.available or ml.explainer is None:
        return templates.TemplateResponse(
            request,
            "_explanation_partial.html",
            {"request": request, "explanation": None, "reason": "not_available"},
        )

    row_idx = ml._bill_id_to_row.get(bill_id)
    if row_idx is None:
        return templates.TemplateResponse(
            request,
            "_explanation_partial.html",
            {"request": request, "explanation": None, "reason": "bill_not_found"},
        )

    try:
        row = ml.feature_matrix[row_idx]
        result = ml.explainer.explain_prediction(row, ml.feature_names)
        bill = state.bills_lookup.get(bill_id)
        score = next(
            (s for s in ml.bill_scores if s.bill_id == bill_id),
            None,
        )
        _enrich_explanation_factors(result, bill_id, bill, score)
        return templates.TemplateResponse(
            request,
            "_explanation_partial.html",
            {"request": request, "explanation": result, "reason": None},
        )
    except Exception:
        _LOGGER.exception("SHAP explanation failed for bill %s", bill_id)
        return templates.TemplateResponse(
            request,
            "_explanation_partial.html",
            {"request": request, "explanation": None, "reason": "error"},
        )
