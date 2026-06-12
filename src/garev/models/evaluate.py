"""Metrics and explainability for the two-stage model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    roc_auc_score,
)

from garev.models.two_stage import TwoStageRevenueModel


@dataclass(frozen=True)
class EvaluationReport:
    """Headline metrics for one evaluation run."""

    n_visitors: int
    n_buyers: int
    classifier_auc: float
    classifier_average_precision: float
    revenue_rmse: float

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(
    model: TwoStageRevenueModel,
    features: pd.DataFrame,
    is_buyer: pd.Series,
    log_revenue: pd.Series,
) -> EvaluationReport:
    """Score a fitted model on held-out data.

    RMSE is computed on the competition target (expected log-revenue), while the
    classifier is judged with AUC and average precision because the positive class
    is rare and accuracy would be meaningless.
    """
    proba = model.predict_proba(features)
    predicted_log_revenue = model.predict_log_revenue(features)

    # AUC/AP need both classes present; guard the degenerate single-class case.
    has_both_classes = is_buyer.nunique() > 1
    auc = roc_auc_score(is_buyer, proba) if has_both_classes else float("nan")
    ap = average_precision_score(is_buyer, proba) if has_both_classes else float("nan")
    rmse = float(np.sqrt(mean_squared_error(log_revenue, predicted_log_revenue)))

    return EvaluationReport(
        n_visitors=len(features),
        n_buyers=int(is_buyer.sum()),
        classifier_auc=float(auc),
        classifier_average_precision=float(ap),
        revenue_rmse=rmse,
    )


def feature_importance(model: TwoStageRevenueModel, top_n: int = 20) -> pd.DataFrame:
    """Return the classifier's gain-based feature importance, most important first."""
    booster = model.classifier
    importance = pd.DataFrame(
        {
            "feature": booster.feature_name_,
            "gain": booster.booster_.feature_importance(importance_type="gain"),
        }
    )
    return importance.sort_values("gain", ascending=False).head(top_n).reset_index(drop=True)
