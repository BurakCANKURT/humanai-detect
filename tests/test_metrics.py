"""evaluation.metrics birim testleri."""

from __future__ import annotations

import math

import numpy as np
import pytest

from humanai_detect.evaluation.metrics import (
    compute_metrics,
    compute_metrics_by_generator,
    format_generator_report,
    format_metrics_report,
)


class TestComputeMetrics:
    _y_true = [0, 1, 2, 0, 1, 2]
    _y_pred = [0, 1, 1, 0, 2, 2]

    def test_accuracy_in_range(self):
        result = compute_metrics(self._y_true, self._y_pred)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_perfect_prediction(self):
        result = compute_metrics([0, 1, 2], [0, 1, 2])
        assert math.isclose(result["accuracy"], 1.0)
        assert math.isclose(result["macro_f1"], 1.0)

    def test_keys_present(self):
        result = compute_metrics(self._y_true, self._y_pred)
        for key in ("accuracy", "macro_f1", "weighted_f1", "per_class_f1", "confusion_matrix"):
            assert key in result

    def test_confusion_matrix_shape(self):
        result = compute_metrics(self._y_true, self._y_pred)
        cm = result["confusion_matrix"]
        assert len(cm) == 3 and len(cm[0]) == 3

    def test_per_class_f1_has_three_classes(self):
        result = compute_metrics(self._y_true, self._y_pred, label_names=["h", "a", "ai"])
        assert len(result["per_class_f1"]) == 3

    def test_roc_auc_nan_without_proba(self):
        result = compute_metrics(self._y_true, self._y_pred)
        assert math.isnan(result["roc_auc_ovr"])

    def test_roc_auc_with_proba(self):
        rng = np.random.default_rng(0)
        proba = rng.dirichlet(np.ones(3), size=len(self._y_true))
        result = compute_metrics(self._y_true, self._y_pred, y_proba=proba)
        assert math.isfinite(result["roc_auc_ovr"])


class TestFormatMetricsReport:
    def test_returns_string(self):
        metrics = compute_metrics([0, 1, 2], [0, 1, 2])
        report = format_metrics_report(metrics)
        assert isinstance(report, str)

    def test_contains_accuracy(self):
        metrics = compute_metrics([0, 1, 2], [0, 2, 1])
        report = format_metrics_report(metrics)
        assert "Accuracy" in report or "accuracy" in report.lower()


class TestComputeMetricsByGenerator:
    _sample_ids = [
        "human_rechunked_0000", "human_rechunked_0001",
        "ai_raw_transformers_0001", "ai_raw_transformers_0002",
        "ai_raw_openai_0001", "ai_raw_openai_0002",
        "ai_raw_anthropic_0001",
    ]
    _y_true = [0, 0, 1, 1, 1, 1, 1]
    _y_pred = [0, 0, 1, 0, 1, 1, 0]  # qwen: 1/2 dogru, gpt: 2/2 dogru, claude: 0/1 dogru

    def test_splits_by_generator(self):
        result = compute_metrics_by_generator(self._y_true, self._y_pred, self._sample_ids)
        assert set(result.keys()) == {"human", "qwen", "gpt4o_mini", "claude_sonnet5"}

    def test_per_generator_n_matches(self):
        result = compute_metrics_by_generator(self._y_true, self._y_pred, self._sample_ids)
        assert result["human"]["n"] == 2
        assert result["qwen"]["n"] == 2
        assert result["gpt4o_mini"]["n"] == 2
        assert result["claude_sonnet5"]["n"] == 1

    def test_per_generator_accuracy_reflects_subset(self):
        result = compute_metrics_by_generator(self._y_true, self._y_pred, self._sample_ids)
        assert result["human"]["accuracy"] == 1.0
        assert result["gpt4o_mini"]["accuracy"] == 1.0
        assert result["claude_sonnet5"]["accuracy"] == 0.0

    def test_format_generator_report_is_string(self):
        result = compute_metrics_by_generator(self._y_true, self._y_pred, self._sample_ids)
        report = format_generator_report(result)
        assert isinstance(report, str)
        assert "claude_sonnet5" in report
