"""SHAP-based prediction explainer for bill outcome models.

Provides human-readable, per-bill explanations of *why* the model
assigned a particular probability.  Uses ``shap.TreeExplainer`` on the
raw (uncalibrated) ``GradientBoostingClassifier`` and converts the
native log-odds SHAP values into probability-space percentage impacts.

Key features:
- One-time ``TreeExplainer`` initialisation at startup (reused for all
  requests -- no per-call overhead).
- Categorical grouping: one-hot encoded columns (e.g.
  ``sponsor_party_democrat``, ``sponsor_party_republican``) are summed
  into a single ``sponsor_party`` master feature before ranking.
- Log-odds to probability conversion via ``scipy.special.expit``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.special import expit

from .features import CATEGORICAL_PREFIXES, humanize_feature_name

LOGGER = logging.getLogger(__name__)

# Number of top drivers to surface in each direction.
TOP_N = 3


class SHAPExplainer:
    """Wraps ``shap.TreeExplainer`` for on-demand bill explanations.

    Instantiated once at app startup and stored in ``MLData``.
    """

    def __init__(self, raw_model: Any) -> None:
        import shap

        self._model = raw_model
        self._explainer = shap.TreeExplainer(raw_model)
        # expected_value can be: a Python scalar, a 1-element numpy array
        # (binary GBM produces a single log-odds margin), or a 2-element
        # array (per-class).  Extract the class-1 value in all cases.
        ev = self._explainer.expected_value
        if np.isscalar(ev):
            self._expected_value = float(ev)
        elif hasattr(ev, "__len__") and len(ev) > 1:
            # Per-class array: take class 1
            self._expected_value = float(ev[1])
        else:
            # 1-element array or 0-d array
            self._expected_value = float(np.asarray(ev).item())
        LOGGER.info(
            "SHAPExplainer initialised (base log-odds=%.4f, base prob=%.2f%%)",
            self._expected_value,
            expit(self._expected_value) * 100,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_prediction(
        self,
        bill_features: np.ndarray | Any,
        feature_names: list[str],
    ) -> dict:
        """Compute SHAP explanation for a single bill.

        Parameters
        ----------
        bill_features:
            1-D array (or single-row sparse matrix) of feature values
            for the bill.
        feature_names:
            Ordered list of feature column names matching the array.

        Returns
        -------
        dict with keys ``base_value``, ``top_positive_factors``,
        ``top_negative_factors``.
        """
        # Convert to dense float64 2-D array — SHAP's TreeExplainer
        # calls np.isnan() which requires a numeric dtype, but the
        # sparse feature matrix can be dtype('O') when TF-IDF and
        # tabular columns are hstacked with mixed types.
        from scipy import sparse as sp

        if sp.issparse(bill_features):
            row = np.asarray(bill_features.todense(), dtype=np.float64)
        else:
            row = np.asarray(bill_features, dtype=np.float64)
        if row.ndim == 1:
            row = row.reshape(1, -1)

        # Raw SHAP values (log-odds space)
        shap_values = self._explainer.shap_values(row)
        if isinstance(shap_values, list):
            # Binary classification: take class-1 values
            shap_values = shap_values[1]
        shap_values = np.asarray(shap_values).flatten()

        # ── Log-odds → probability-space impacts ──────────────────────
        current_margin = self._expected_value + float(shap_values.sum())
        current_prob = float(expit(current_margin))

        raw_shap_dict: dict[str, float] = {}
        for name, sv in zip(feature_names, shap_values):
            margin_without = current_margin - float(sv)
            impact_pct = current_prob - float(expit(margin_without))
            raw_shap_dict[name] = impact_pct

        # ── Group one-hot categoricals ────────────────────────────────
        grouped = _group_categorical_features(raw_shap_dict)

        # ── Rank by absolute magnitude ────────────────────────────────
        positives = sorted(
            ((k, v) for k, v in grouped.items() if v > 0),
            key=lambda kv: kv[1],
            reverse=True,
        )
        negatives = sorted(
            ((k, v) for k, v in grouped.items() if v < 0),
            key=lambda kv: kv[1],
        )

        def _fmt(items: list[tuple[str, float]], limit: int = TOP_N) -> list[dict]:
            results = []
            for raw_name, impact in items[:limit]:
                human = humanize_feature_name(raw_name)
                pct = impact * 100
                sign = "+" if pct >= 0 else ""
                results.append(
                    {
                        "feature": human,
                        "impact": f"{sign}{pct:.1f}%",
                        "raw_impact": round(impact, 6),
                        "raw_feature": raw_name,
                    }
                )
            return results

        base_prob = float(expit(self._expected_value))

        return {
            "base_value": round(base_prob, 4),
            "top_positive_factors": _fmt(positives),
            "top_negative_factors": _fmt(negatives),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _group_categorical_features(shap_dict: dict[str, float]) -> dict[str, float]:
    """Sum one-hot encoded dummy columns into master features.

    For every prefix in ``CATEGORICAL_PREFIXES``, columns that start with
    the prefix (e.g. ``sponsor_party_democrat``) are summed into a single
    entry keyed by the prefix *without* the trailing underscore
    (``sponsor_party``).
    """
    grouped: dict[str, float] = {}
    consumed: set[str] = set()

    for prefix in CATEGORICAL_PREFIXES:
        master_key = prefix.rstrip("_")
        total = 0.0
        for key, val in shap_dict.items():
            if key.startswith(prefix):
                total += val
                consumed.add(key)
        if total != 0.0 or any(k.startswith(prefix) for k in shap_dict):
            grouped[master_key] = total

    for key, val in shap_dict.items():
        if key not in consumed:
            grouped[key] = val

    return grouped
