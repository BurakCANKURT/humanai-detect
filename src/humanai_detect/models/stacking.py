"""Temel modellerden sklearn StackingClassifier ensemble olusturma."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import StackingClassifier

from .factory import build_model


def build_stacking_ensemble(
    base_model_configs: list[tuple[str, dict[str, Any]]],
    meta_learner_config: tuple[str, dict[str, Any]],
    passthrough: bool = False,
    cv: int = 5,
) -> StackingClassifier:
    """Temel modelleri ve meta-learner'i alip sklearn StackingClassifier dondurur.

    base_model_configs : [(model_name, params_dict), ...]
    meta_learner_config: (model_name, params_dict) — genellikle logreg
    passthrough        : True -> orijinal ozellikler de meta-learner'a iletilir
    cv                 : base model tahminleri uretilirken kullanilacak fold sayisi

    Kullanim:
        base_cfgs = [("xgboost", xgb_params), ("catboost", cat_params)]
        meta_cfg  = ("logreg", lr_params)
        stacker   = build_stacking_ensemble(base_cfgs, meta_cfg)
        stacker.fit(X_train, y_train)
    """
    estimators = [
        (name, build_model(name, params))
        for name, params in base_model_configs
    ]
    meta_name, meta_params = meta_learner_config
    final_estimator = build_model(meta_name, meta_params)

    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        passthrough=passthrough,
        cv=cv,
        n_jobs=-1,
    )


def build_stacking_from_config(models_cfg: dict[str, Any]) -> StackingClassifier:
    """configs/models.yaml'dan okunmak uzere stacking konfigurasyonunu isle."""
    stacking_cfg = models_cfg.get("stacking", {})
    common = models_cfg.get("common", {})

    base_names: list[str] = stacking_cfg.get("base_models", ["xgboost", "catboost", "mlp", "logreg"])
    meta_name: str = stacking_cfg.get("meta_learner", "logreg")
    passthrough: bool = stacking_cfg.get("passthrough", False)

    base_cfgs = [
        (name, {**common, **models_cfg.get(name, {})})
        for name in base_names
    ]
    meta_cfg = (meta_name, {**common, **models_cfg.get(meta_name, {})})

    return build_stacking_ensemble(base_cfgs, meta_cfg, passthrough=passthrough)
