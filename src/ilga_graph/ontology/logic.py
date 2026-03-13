"""Logic layer: computed property helpers and alert monitor over the ontology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..app_state import AppState


@dataclass
class LogicAlert:
    """A single alert produced by the LogicMonitor."""

    trigger: str
    object_type: str
    object_id: str
    message: str
    severity: str = "medium"


class LogicMonitor:
    """Watches ontology state and produces alerts when thresholds are crossed.

    Checks:
    - Witness slip anomalies (ML): bills flagged as astroturfing risk.
    - High-controversy open bills: many opponent slips, bill still in progress.
    - High-influence legislator with recent sponsorship (optional).
    """

    def __init__(self, state: AppState) -> None:
        self._state = state

    def get_alerts(self) -> list[LogicAlert]:
        """Return current alerts from anomaly detection and controversy heuristics."""
        alerts: list[LogicAlert] = []
        ml = getattr(self._state, "ml", None)
        if ml and getattr(ml, "anomalies", None):
            for a in ml.anomalies:
                if getattr(a, "is_anomaly", False):
                    alerts.append(
                        LogicAlert(
                            trigger="slip_anomaly",
                            object_type="bill",
                            object_id=getattr(a, "bill_id", "") or getattr(a, "bill_number", ""),
                            message=getattr(a, "anomaly_reason", "")
                            or "Unusual witness slip activity",
                            severity="high",
                        )
                    )
        bill_lookup = getattr(self._state, "bills_lookup", None) or {}
        slips_lookup = getattr(self._state, "witness_slips_lookup", None) or {}
        for leg_id, bill in list(bill_lookup.items())[:500]:
            slip_list = slips_lookup.get(getattr(bill, "bill_number", ""), []) or []
            if len(slip_list) < 5:
                continue
            opp = sum(
                1 for s in slip_list if (getattr(s, "position", "") or "").lower() == "opponent"
            )
            pro = sum(
                1 for s in slip_list if (getattr(s, "position", "") or "").lower() == "proponent"
            )
            total = opp + pro
            if total and (opp / total) > 0.6:
                last = getattr(bill, "last_action", "") or ""
                if "Public Act" not in last and "Signed" not in last:
                    alerts.append(
                        LogicAlert(
                            trigger="high_controversy",
                            object_type="bill",
                            object_id=leg_id,
                            message=f"Open bill with {opp} opponent vs {pro} proponent slips",
                            severity="medium",
                        )
                    )
        return alerts
