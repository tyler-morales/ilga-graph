"""Ontology SDK (OSDK): public API for the application plane to consume ontology objects."""

from __future__ import annotations

from typing import Any

from .actions import ActionResult, OntologyAction
from .logic import LogicAlert, LogicMonitor
from .mapping import MappingService
from .objects import (
    BaseOntologyObject,
    BillObject,
    LegislatorObject,
)


class OntologySDK:
    """Single public API for querying objects, traversing links, executing actions, and logic."""

    def __init__(self, mapping: MappingService) -> None:
        self._mapping = mapping

    def get_legislator(self, id: str) -> LegislatorObject | None:
        return self._mapping.hydrate_legislator(id)

    def get_bill(self, leg_id: str) -> BillObject | None:
        return self._mapping.hydrate_bill(leg_id)

    def get_committee(self, code: str) -> Any:
        return self._mapping.hydrate_committee(code)

    def get_organization(self, canonical_name: str) -> Any:
        return self._mapping.hydrate_organization(canonical_name)

    def search(
        self,
        query: str,
        types: list[str] | None = None,
    ) -> list[BaseOntologyObject]:
        q = (query or "").strip().lower()
        if not q:
            return []
        types = types or ["legislator", "bill", "committee"]
        results: list[BaseOntologyObject] = []
        if "legislator" in types:
            for m in self._mapping._state.members or []:
                if q in (m.name or "").lower() or q in (m.district or "").lower():
                    obj = self._mapping.hydrate_legislator(m.id)
                    if obj:
                        results.append(obj)
        if "bill" in types:
            for b in self._mapping._state.bills or []:
                leg_id = getattr(b, "leg_id", None) or ""
                bn = getattr(b, "bill_number", "") or ""
                desc = (getattr(b, "description", "") or "").lower()
                if q in bn.lower() or q in desc or q in leg_id:
                    obj = self._mapping.hydrate_bill(leg_id)
                    if obj and obj not in results:
                        results.append(obj)
        if "committee" in types:
            for c in self._mapping._state.committees or []:
                code = getattr(c, "code", "") or ""
                name = (getattr(c, "name", "") or "").lower()
                if q in code.lower() or q in name:
                    obj = self._mapping.hydrate_committee(code)
                    if obj:
                        results.append(obj)
        return results

    def linked_objects(
        self,
        obj: BaseOntologyObject,
        link_type: str,
    ) -> list[BaseOntologyObject]:
        out: list[BaseOntologyObject] = []
        for link in obj.links:
            if link.link_type != link_type:
                continue
            tid, ttype = link.target_id, link.target_type
            if ttype == "legislator":
                o = self._mapping.hydrate_legislator(tid)
            elif ttype == "bill":
                o = self._mapping.hydrate_bill(tid)
            elif ttype == "committee":
                o = self._mapping.hydrate_committee(tid)
            elif ttype == "organization":
                o = self._mapping.hydrate_organization(tid)
            else:
                o = None
            if o is not None:
                out.append(o)
        return out

    def neighbors(
        self,
        obj: BaseOntologyObject,
        depth: int = 1,
    ) -> list[BaseOntologyObject]:
        if depth < 1:
            return []
        seen: set[str] = {obj.object_id}
        frontier: list[BaseOntologyObject] = [obj]
        accumulated: list[BaseOntologyObject] = []
        for _ in range(depth):
            next_frontier: list[BaseOntologyObject] = []
            for node in frontier:
                for link in node.links:
                    tid, ttype = link.target_id, link.target_type
                    if tid in seen:
                        continue
                    seen.add(tid)
                    o = None
                    if ttype == "legislator":
                        o = self._mapping.hydrate_legislator(tid)
                    elif ttype == "bill":
                        o = self._mapping.hydrate_bill(tid)
                    elif ttype == "committee":
                        o = self._mapping.hydrate_committee(tid)
                    elif ttype == "organization":
                        o = self._mapping.hydrate_organization(tid)
                    if o is not None:
                        next_frontier.append(o)
                        accumulated.append(o)
            frontier = next_frontier
        return accumulated

    def execute_action(self, action: OntologyAction) -> ActionResult:
        """Record an ontology action (e.g. after outreach). Appends to in-memory log."""
        state = self._mapping._state
        if not hasattr(state, "ontology_actions"):
            state.ontology_actions = []
        state.ontology_actions.append(action)
        return ActionResult(success=True, action_id=action.action_id, message="Recorded")

    def action_history(self, target_id: str) -> list[OntologyAction]:
        """Return actions that targeted the given object. From in-memory log."""
        state = self._mapping._state
        actions = getattr(state, "ontology_actions", []) or []
        return [
            a for a in actions
            if any(lnk.target_id == target_id for lnk in a.target_links)
        ]

    def predict_bill(self, leg_id: str) -> dict[str, Any] | None:
        """Return bill prediction from ML pipeline if available."""
        state = self._mapping._state
        if not getattr(state, "ml", None) or not getattr(state.ml, "bill_scores", None):
            return None
        for s in state.ml.bill_scores:
            if getattr(s, "bill_id", None) == leg_id:
                return {
                    "prob_advance": getattr(s, "prob_advance", None),
                    "prob_law": getattr(s, "prob_law", None),
                    "predicted_outcome": getattr(s, "predicted_outcome", None),
                    "confidence": getattr(s, "confidence", None),
                }
        bill = state.bills_lookup.get(leg_id)
        if bill and getattr(bill, "bill_number", None):
            for s in state.ml.bill_scores:
                if getattr(s, "bill_number", None) == bill.bill_number:
                    return {
                        "prob_advance": getattr(s, "prob_advance", None),
                        "prob_law": getattr(s, "prob_law", None),
                        "predicted_outcome": getattr(s, "predicted_outcome", None),
                        "confidence": getattr(s, "confidence", None),
                    }
        return None

    def get_alerts(self) -> list[LogicAlert]:
        """Return logic-layer alerts (anomalies, high-controversy bills)."""
        return LogicMonitor(self._mapping._state).get_alerts()

    def influence_rank(self, chamber: str | None = None) -> list[LegislatorObject]:
        """Return legislators ordered by influence score (high to low)."""
        state = self._mapping._state
        inf = getattr(state, "influence", None) or {}
        if not inf:
            return []
        order = sorted(
            inf.keys(),
            key=lambda mid: getattr(inf.get(mid), "influence_score", 0) or 0,
            reverse=True,
        )
        out: list[LegislatorObject] = []
        for mid in order:
            prof = inf.get(mid)
            if chamber and prof and getattr(prof, "chamber", "") != chamber:
                continue
            obj = self._mapping.hydrate_legislator(mid)
            if obj:
                out.append(obj)
        return out
