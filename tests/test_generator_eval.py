"""evaluation.generator_eval birim testleri (model egitimi gerektirmeyen yardimci fonksiyonlar).

run_logo_cv gercek stacking egitimi gerektirdigi icin (yavas, olcum/tani script'i)
kapsam disi birakildi -- explain.py/measure_calibration.py gibi agir script'ler
de mevcut suite'te dogrudan test edilmiyor, ayni konvansiyon."""

from __future__ import annotations

import pandas as pd

from humanai_detect.evaluation.generator_eval import (
    add_generator_column,
    balance_by_label,
    build_logo_fold,
    split_human,
)


def _toy_df() -> pd.DataFrame:
    rows = []
    for i in range(10):
        rows.append({"sample_id": f"human_rechunked_{i:04d}", "label": "human", "feat1": float(i)})
    for i in range(6):
        rows.append({"sample_id": f"ai_raw_transformers_{i:04d}", "label": "ai_raw", "feat1": float(i)})
    for i in range(4):
        rows.append({"sample_id": f"ai_raw_openai_{i:04d}", "label": "ai_raw", "feat1": float(i)})
    for i in range(2):
        rows.append({"sample_id": f"ai_raw_anthropic_{i:04d}", "label": "ai_raw", "feat1": float(i)})
    for i in range(5):
        rows.append({"sample_id": f"ai_humanized_backtranslate_ai_raw_transformers_{i:04d}",
                      "label": "ai_humanized", "feat1": float(i)})
    return pd.DataFrame(rows)


class TestAddGeneratorColumn:
    def test_adds_column(self):
        df = add_generator_column(_toy_df())
        assert "generator" in df.columns

    def test_generator_values_correct(self):
        df = add_generator_column(_toy_df())
        counts = df["generator"].value_counts().to_dict()
        assert counts["human"] == 10
        assert counts["qwen"] == 6 + 5
        assert counts["gpt4o_mini"] == 4
        assert counts["claude_sonnet5"] == 2

    def test_idempotent_if_already_present(self):
        df = add_generator_column(_toy_df())
        df2 = add_generator_column(df)
        assert df2["generator"].equals(df["generator"])


class TestSplitHuman:
    def test_train_test_disjoint(self):
        df = _toy_df()
        train, test = split_human(df, test_frac=0.3, random_state=42)
        assert set(train.index).isdisjoint(set(test.index))

    def test_only_human_rows(self):
        df = _toy_df()
        train, test = split_human(df, test_frac=0.3, random_state=42)
        assert (train["label"] == "human").all()
        assert (test["label"] == "human").all()

    def test_covers_all_human_rows(self):
        df = _toy_df()
        train, test = split_human(df, test_frac=0.3, random_state=42)
        assert len(train) + len(test) == (df["label"] == "human").sum()


class TestBuildLogoFold:
    def test_held_out_generator_absent_from_train(self):
        df = add_generator_column(_toy_df())
        train_human, test_human = split_human(df, random_state=42)
        train_df, test_df = build_logo_fold(df, "qwen", train_human, test_human)
        assert (train_df["generator"] != "qwen").all()

    def test_held_out_generator_present_in_test(self):
        df = add_generator_column(_toy_df())
        train_human, test_human = split_human(df, random_state=42)
        train_df, test_df = build_logo_fold(df, "qwen", train_human, test_human)
        assert (test_df["generator"] == "qwen").sum() == 11  # 6 ai_raw + 5 ai_humanized

    def test_other_generators_stay_in_train(self):
        df = add_generator_column(_toy_df())
        train_human, test_human = split_human(df, random_state=42)
        train_df, test_df = build_logo_fold(df, "qwen", train_human, test_human)
        assert (train_df["generator"] == "gpt4o_mini").sum() == 4
        assert (train_df["generator"] == "claude_sonnet5").sum() == 2


class TestBalanceByLabel:
    def test_equal_counts_per_label(self):
        df = _toy_df()
        balanced = balance_by_label(df, random_state=42)
        counts = balanced["label"].value_counts()
        assert counts.nunique() == 1

    def test_matches_min_class_count(self):
        df = _toy_df()
        min_n = df["label"].value_counts().min()
        balanced = balance_by_label(df, random_state=42)
        assert (balanced["label"].value_counts() == min_n).all()
