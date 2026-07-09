"""Config'ten model adi + hiperparametre alip sklearn-uyumlu estimator ureten factory."""

from __future__ import annotations

from typing import Any


def build_model(name: str, params: dict[str, Any]):
    """name (xgboost|catboost|mlp|logreg) icin yapilandirilmis estimator dondurur.

    params, configs/models.yaml'in ilgili alt sozlugunun ustune override edilebilecek
    hiperparametreleri icerir; eksik degerler YAML varsayilanlariyla tamamlanir.
    """
    rs = params.get("random_state", 42)
    cw = params.get("class_weight", "balanced")

    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            max_depth=params.get("max_depth", 6),
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("eta", 0.1),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 1),
            random_state=rs,
            eval_metric="mlogloss",
            verbosity=0,
        )

    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            depth=params.get("depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            iterations=params.get("iterations", 500),
            random_seed=rs,
            auto_class_weights="Balanced" if cw == "balanced" else None,
            verbose=0,
        )

    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        # sklearn MLPClassifier dropout desteklemiyor; early_stopping benzer etki saglar.
        return MLPClassifier(
            hidden_layer_sizes=tuple(params.get("hidden_layers", [256, 128, 64])),
            activation=params.get("activation", "relu"),
            max_iter=params.get("epochs", 50),
            batch_size=params.get("batch_size", "auto"),
            learning_rate_init=params.get("learning_rate", 0.001),
            early_stopping=True,
            validation_fraction=0.1,
            random_state=rs,
        )

    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        penalty = params.get("penalty", "l2")
        C = params.get("C", 1.0)
        # sklearn 1.8+: penalty kaldirildi; l1_ratio ile L1/L2 kontrolu yapiliyor
        if penalty == "l1":
            return LogisticRegression(
                l1_ratio=1.0, C=C,
                max_iter=params.get("max_iter", 1000),
                solver="saga",
                class_weight=cw, random_state=rs,
            )
        # l2 (varsayilan)
        return LogisticRegression(
            l1_ratio=0.0, C=C,
            max_iter=params.get("max_iter", 1000),
            solver="lbfgs",
            class_weight=cw, random_state=rs,
        )

    if name == "stacking":
        from .stacking import build_stacking_from_config

        # params burada tekil model params degil, configs/models.yaml'in tamami
        # (common + stacking + xgboost/catboost/mlp/logreg alt sozlukleri) olmali.
        return build_stacking_from_config(params)

    raise ValueError(f"Bilinmeyen model adi: {name!r}  (xgboost | catboost | mlp | logreg | stacking)")
