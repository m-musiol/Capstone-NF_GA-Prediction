"""Two-stage revenue model: classify return-buyers, then regress their revenue.

Because almost every visitor produces zero future revenue, a single regressor
wastes its capacity modelling the zeros. The competition-proven structure splits
the problem:

1. a **classifier** estimates ``P(visitor returns and buys)``;
2. a **regressor** estimates the log-revenue assuming a purchase happens.

The expected log-revenue used for scoring is the product of the two. Both stages
are LightGBM models trained with early stopping on a held-out validation split.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from garev.config import ModelConfig


def _common_params(config: ModelConfig) -> dict:
    """LightGBM hyper-parameters shared by both stages."""
    return {
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
        "random_state": config.random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }


@dataclass
class TwoStageRevenueModel:
    """Wraps the classifier and regressor behind a single fit/predict API."""

    config: ModelConfig

    def __post_init__(self) -> None:
        params = _common_params(self.config)
        self.classifier = lgb.LGBMClassifier(objective="binary", **params)
        self.regressor = lgb.LGBMRegressor(objective="regression", **params)

    def _fit_stage(self, estimator, features: pd.DataFrame, target: pd.Series, metric: str):
        """Fit one stage with an early-stopping validation split."""
        x_train, x_valid, y_train, y_valid = train_test_split(
            features,
            target,
            test_size=self.config.valid_fraction,
            random_state=self.config.random_state,
        )
        estimator.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric=metric,
            callbacks=[lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)],
        )
        return estimator

    def fit(
        self,
        features: pd.DataFrame,
        is_buyer: pd.Series,
        log_revenue: pd.Series,
    ) -> TwoStageRevenueModel:
        """Train both stages.

        The regressor learns only from rows that actually produced revenue, so it
        models *purchase size* rather than being swamped by zeros.
        """
        self._fit_stage(self.classifier, features, is_buyer, metric="binary_logloss")

        buyer_mask = is_buyer.astype(bool)
        if buyer_mask.sum() >= 2:
            self._fit_stage(
                self.regressor, features[buyer_mask], log_revenue[buyer_mask], metric="rmse"
            )
        else:  # Degenerate sample with no buyers: fall back to a zero regressor.
            self.regressor.fit(features, log_revenue)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return ``P(return-buyer)`` for each visitor."""
        return self.classifier.predict_proba(features)[:, 1]

    def predict_log_revenue(self, features: pd.DataFrame) -> np.ndarray:
        """Return expected log-revenue = P(buyer) x predicted log-revenue."""
        proba = self.predict_proba(features)
        conditional = np.clip(self.regressor.predict(features), 0, None)
        return proba * conditional
