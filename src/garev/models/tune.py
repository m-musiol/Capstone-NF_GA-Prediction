"""Optional Optuna hyper-parameter search for the classifier stage.

Optuna is an optional dependency (``pip install garev[tune]``); it is imported
lazily so the core pipeline has no hard dependency on it. The search optimises the
classifier's validation AUC, which drives the ranking of return-buyers — the part
of the problem where extra tuning pays off most.
"""

from __future__ import annotations

from dataclasses import replace

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from garev.config import ModelConfig


def tune_classifier(
    features: pd.DataFrame,
    is_buyer: pd.Series,
    base_config: ModelConfig,
    n_trials: int = 30,
) -> ModelConfig:
    """Search LightGBM hyper-parameters and return an updated ``ModelConfig``.

    Raises:
        ImportError: if Optuna is not installed.
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("Optuna is required for tuning: pip install 'garev[tune]'") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        is_buyer,
        test_size=base_config.valid_fraction,
        random_state=base_config.random_state,
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=base_config.n_estimators,
            random_state=base_config.random_state,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(base_config.early_stopping_rounds, verbose=False)],
        )
        return roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return replace(base_config, **study.best_params)
