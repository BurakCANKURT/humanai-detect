"""Optuna ile hiperparametre optimizasyonu (3-fold CV, macro-F1 hedef metrigi)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .factory import build_model


def _suggest_params(trial: Any, model_name: str) -> dict[str, Any]:
    """Her model icin Optuna'nin ornekleyecegi hiperparametre uzayini tanimlar."""
    if model_name == "xgboost":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
    if model_name == "catboost":
        return {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 200, 1000),
        }
    if model_name == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 4)
        hidden_layers = [trial.suggest_int(f"n_units_l{i}", 64, 512) for i in range(n_layers)]
        return {
            "hidden_layers": hidden_layers,
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 30, 100),
        }
    if model_name == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        }
    raise ValueError(f"HPO icin bilinmeyen model: {model_name!r}")


def run_optuna_study(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    metric: str = "macro_f1",
    cv_folds: int = 3,
    timeout: int | None = None,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Belirtilen model icin Optuna study calistirir, en iyi parametreleri dondurur.

    metric: 'macro_f1' | 'accuracy' | 'weighted_f1'
    timeout: saniye cinsinden maksimum sure (None = sinirsiz)
    groups: verilirse StratifiedGroupKFold kullanilir (kaynak-duzeyinde leakage onleme)

    Donus: {'best_params': dict, 'best_value': float, 'n_trials': int}
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if groups is not None:
        skf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        splits = list(skf.split(X, y, groups=groups))
    else:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        splits = list(skf.split(X, y))

    def objective(trial: Any) -> float:
        params = _suggest_params(trial, model_name)
        scores: list[float] = []
        for train_idx, val_idx in splits:
            model = build_model(model_name, params)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
            if metric == "macro_f1":
                scores.append(f1_score(y[val_idx], y_pred, average="macro", zero_division=0))
            elif metric == "weighted_f1":
                scores.append(f1_score(y[val_idx], y_pred, average="weighted", zero_division=0))
            else:
                from sklearn.metrics import accuracy_score
                scores.append(accuracy_score(y[val_idx], y_pred))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}_hpo",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    trials_data = [
        {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
        for t in study.trials
    ]
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "trials": trials_data,
    }
