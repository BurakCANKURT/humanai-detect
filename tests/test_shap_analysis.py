"""explainability.shap_analysis birim testleri (sadece kalibrasyon-soyma yardimcisi --
SHAP'in kendisi agir/yavas oldugu icin (bkz. proje notlari, TreeExplainer bile buyuk
veride 25+ dakika) mevcut suite'te dogrudan test edilmiyor, ayni konvansiyon
run_shap_global/run_shap_local icin de gecerli)."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

from humanai_detect.explainability.shap_analysis import _unwrap_calibrated_estimator


def _toy_xy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    y = rng.integers(0, 2, size=30)
    return X, y


class TestUnwrapCalibratedEstimator:
    def test_passes_through_non_calibrated_model(self):
        model = DecisionTreeClassifier()
        assert _unwrap_calibrated_estimator(model) is model

    def test_unwraps_calibrated_classifier_cv(self):
        X, y = _toy_xy()
        base = DecisionTreeClassifier(random_state=0)
        calibrated = CalibratedClassifierCV(base, cv=3)
        calibrated.fit(X, y)

        unwrapped = _unwrap_calibrated_estimator(calibrated)

        # TreeExplainer'in tanidigi bir agac tipi olmali, CalibratedClassifierCV DEGIL
        assert type(unwrapped).__name__ != "CalibratedClassifierCV"
        assert isinstance(unwrapped, DecisionTreeClassifier)

    def test_unwrapped_estimator_is_fitted(self):
        X, y = _toy_xy()
        base = DecisionTreeClassifier(random_state=0)
        calibrated = CalibratedClassifierCV(base, cv=3)
        calibrated.fit(X, y)

        unwrapped = _unwrap_calibrated_estimator(calibrated)

        # egitilmis bir estimator predict yapabilmeli (NotFittedError firlatmamali)
        preds = unwrapped.predict(X)
        assert len(preds) == len(y)
