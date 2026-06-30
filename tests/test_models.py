"""Asama 6 model modullerinin birim testleri.

Agir modeller (XGBoost, CatBoost) kucuk sentetik veriyle test edilir.
HPO ve stacking testleri minimum trial sayisiyla calistirilir.
"""

from __future__ import annotations

import numpy as np
import pytest

# Kucuk 3-sinifli sentetik veri seti
_N, _D = 60, 10
_RNG = np.random.default_rng(42)
_X = _RNG.standard_normal((_N, _D)).astype(np.float32)
_Y = np.array([0] * 20 + [1] * 20 + [2] * 20)


# ---------------------------------------------------------------------------
# factory.py
# ---------------------------------------------------------------------------

class TestBuildModel:
    @pytest.mark.parametrize("name", ["xgboost", "catboost", "mlp", "logreg"])
    def test_builds_without_error(self, name):
        from humanai_detect.models.factory import build_model
        model = build_model(name, {})
        assert model is not None

    @pytest.mark.parametrize("name", ["xgboost", "catboost", "mlp", "logreg"])
    def test_fit_predict(self, name):
        from humanai_detect.models.factory import build_model
        model = build_model(name, {"n_estimators": 10, "iterations": 10, "epochs": 5})
        model.fit(_X, _Y)
        preds = np.array(model.predict(_X)).reshape(-1)  # CatBoost (N,1) donebilir
        assert preds.shape == (_N,)
        assert set(preds.tolist()).issubset({0, 1, 2})

    def test_unknown_model_raises(self):
        from humanai_detect.models.factory import build_model
        with pytest.raises(ValueError):
            build_model("unknown_model", {})


# ---------------------------------------------------------------------------
# train.py
# ---------------------------------------------------------------------------

class TestCvTraining:
    def test_returns_fold_metrics(self):
        from humanai_detect.models.train import run_cv_training
        result = run_cv_training(
            _X, _Y, "logreg", {"C": 0.1}, cv_folds=2
        )
        assert "fold_metrics" in result
        assert len(result["fold_metrics"]) == 2

    def test_mean_macro_f1_in_range(self):
        from humanai_detect.models.train import run_cv_training
        result = run_cv_training(_X, _Y, "logreg", {}, cv_folds=3)
        assert 0.0 <= result["mean_macro_f1"] <= 1.0

    def test_train_final_model_saves(self, tmp_path):
        from humanai_detect.models.train import train_final_model
        save_path = tmp_path / "model.pkl"
        model = train_final_model(_X, _Y, "logreg", {}, save_path=save_path)
        assert save_path.exists()
        preds = model.predict(_X)
        assert preds.shape == (_N,)


# ---------------------------------------------------------------------------
# stacking.py
# ---------------------------------------------------------------------------

class TestStackingEnsemble:
    def test_builds_stacking_classifier(self):
        from sklearn.ensemble import StackingClassifier
        from humanai_detect.models.stacking import build_stacking_ensemble
        base_cfgs = [("logreg", {"C": 0.5}), ("logreg", {"C": 1.0})]
        meta_cfg = ("logreg", {})
        stacker = build_stacking_ensemble(base_cfgs, meta_cfg, cv=2)
        assert isinstance(stacker, StackingClassifier)

    def test_stacking_fit_predict(self):
        from humanai_detect.models.stacking import build_stacking_ensemble
        # Isim catismasi olmamasi icin farkli model turleri kullan
        base_cfgs = [("logreg", {}), ("xgboost", {"n_estimators": 10})]
        stacker = build_stacking_ensemble(base_cfgs, ("logreg", {}), cv=2)
        stacker.fit(_X, _Y)
        preds = stacker.predict(_X)
        assert preds.shape == (_N,)

    def test_build_from_config(self):
        from humanai_detect.models.stacking import build_stacking_from_config
        cfg = {
            "common": {"random_state": 42},
            "logreg": {"C": 1.0},
            "stacking": {
                "base_models": ["logreg", "logreg"],
                "meta_learner": "logreg",
                "passthrough": False,
            },
        }
        stacker = build_stacking_from_config(cfg)
        assert stacker is not None
