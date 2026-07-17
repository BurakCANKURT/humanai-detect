"""Canli metin tahmini icin cikarim (inference) katmani.

Egitim pipeline'inin (src/humanai_detect) ayni fonksiyonlarini KULLANIR ama
hicbir dosyasini degistirmez. Akis, scripts/preprocess.py + scripts/build_features.py
ile birebir ayni adimlari tek bir metin uzerinde tekrarlar:

    ham metin -> clean_text -> Stanza analiz (tokens/pos/dep) -> perplexity/burstiness
    -> extract_all_features (stilometri, 35 ozellik) -> BERTurk embedding (768 boyut)
    -> z-score standardizasyon (build_artifacts.py'de onceden hesaplanan mean/std)
    -> egitilmis stacking modeli (outputs/models/stacking_final_production.pkl)

Standardizasyon istatistikleri ve insan-korpusu referans sozlugu (kl-div, dep-depth,
conjunction deviation icin) arayuz/artifacts/ altinda onceden hesaplanip kaydedildi
(bkz. build_artifacts.py) — bunlar egitim sirasinda kullanilanla birebir aynidir.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARAYUZ_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARAYUZ_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from humanai_detect.config import load_yaml  # noqa: E402
from humanai_detect.preprocessing.cleaning import clean_text  # noqa: E402
from humanai_detect.preprocessing.linguistic import analyze  # noqa: E402
from humanai_detect.preprocessing.perplexity import compute_perplexity  # noqa: E402
from humanai_detect.preprocessing.burstiness import compute_burstiness  # noqa: E402
from humanai_detect.preprocessing.schemas import ProcessedSample  # noqa: E402
from humanai_detect.features.aggregator import extract_all_features  # noqa: E402
from humanai_detect.fusion.length_residualize import apply_length_residualizer_dict  # noqa: E402
from humanai_detect.embeddings.berturk import embed_berturk  # noqa: E402

ARTIFACTS_DIR = ARAYUZ_DIR / "artifacts"
LABEL_NAMES = ["human", "ai_raw", "ai_humanized"]
LABEL_TR = {
    "human": "İnsan Yazımı",
    "ai_raw": "Ham Yapay Zeka Üretimi",
    "ai_humanized": "İnsanlaştırılmış Yapay Zeka Üretimi",
}
MODEL_FILE = "stacking_final_production.pkl"

BINARY_LABEL_NAMES = ["human", "ai"]
BINARY_LABEL_TR = {
    "human": "İnsan Yazımı",
    "ai": "Yapay Zeka Üretimi (ham veya insanlaştırılmış, ayrım yapılmıyor)",
}
BINARY_MODEL_FILE = "_diag_after_calibrated_binary.pkl"


@dataclass
class PredictionResult:
    label: str
    label_tr: str
    probabilities: dict[str, float]
    token_count: int
    sentence_count: int
    perplexity: float
    burstiness: float
    raw_features: dict[str, float]
    warnings: list[str]


class InferenceEngine:
    """Model + artefaktlari bir kez yukleyip tekrar tekrar tahmin yapmak icin."""

    def __init__(
        self,
        model_file: str = MODEL_FILE,
        label_names: list[str] = LABEL_NAMES,
        label_tr: dict[str, str] = LABEL_TR,
    ) -> None:
        self._label_names = label_names
        self._label_tr = label_tr
        self._features_cfg = load_yaml("features")
        self._preprocessing_cfg = load_yaml("preprocessing")
        self._emb_cfg = load_yaml("embeddings")

        reference_path = ARTIFACTS_DIR / "reference.json"
        scaler_path = ARTIFACTS_DIR / "scaler.npz"
        feat_cols_path = ARTIFACTS_DIR / "feature_columns.json"
        if not (reference_path.exists() and scaler_path.exists() and feat_cols_path.exists()):
            raise FileNotFoundError(
                "Artefaktlar bulunamadi. Once 'python arayuz/build_artifacts.py' calistirin."
            )

        self._reference = json.loads(reference_path.read_text(encoding="utf-8"))
        scaler = np.load(scaler_path, allow_pickle=True)
        self._sty_cols = list(scaler["sty_cols"])
        self._sty_mean = scaler["sty_mean"]
        self._sty_std = scaler["sty_std"]
        self._emb_mean = scaler["emb_mean"]
        self._emb_std = scaler["emb_std"]
        self._feature_columns = json.loads(feat_cols_path.read_text(encoding="utf-8"))

        length_params_path = ARTIFACTS_DIR / "length_residualizer.json"
        self._length_params = (
            json.loads(length_params_path.read_text(encoding="utf-8"))
            if length_params_path.exists() else {}
        )

        import pickle
        model_path = PROJECT_ROOT / "outputs" / "models" / model_file
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

    def predict(self, raw_text: str) -> PredictionResult:
        warnings: list[str] = []

        cleaned = clean_text(raw_text)
        if not cleaned:
            raise ValueError("Temizleme sonrasi metin bos kaldi.")

        result = analyze(cleaned)
        tokens = result["tokens"]
        token_count = len(tokens)
        sentence_count = len(result["sentences"])

        min_tok = self._preprocessing_cfg["min_tokens"]
        max_tok = self._preprocessing_cfg["max_tokens"]
        if token_count < min_tok:
            warnings.append(
                f"Metin cok kisa ({token_count} token, egitim araligi {min_tok}-{max_tok}). "
                "Tahmin guvenilirligi dusuk olabilir."
            )
        elif token_count > max_tok:
            warnings.append(
                f"Metin cok uzun ({token_count} token, egitim araligi {min_tok}-{max_tok}). "
                "Yalnizca modelin gordugu uzunluk araligi icin guven verilebilir."
            )

        sentence_lengths = [len(s.split()) for s in result["sentences"]]
        burstiness = compute_burstiness(sentence_lengths)
        perplexity = compute_perplexity(cleaned, self._preprocessing_cfg["perplexity_model_id"])

        sample = ProcessedSample(
            id="live",
            text=raw_text,
            cleaned_text=cleaned,
            label="human",  # tahmin amacli placeholder, egitimde kullanilmiyor
            source="live_input",
            sentences=result["sentences"],
            tokens=tokens,
            pos_tags=result["pos_tags"],
            dep_parse=result["dep_parse"],
            token_count=token_count,
            sentence_count=sentence_count,
            perplexity=perplexity,
            burstiness=burstiness,
        )

        feats = extract_all_features(sample, self._features_cfg, reference=self._reference)
        if self._length_params:
            feats = apply_length_residualizer_dict(feats, token_count, self._length_params)
        sty_vec = np.array([feats.get(c, np.nan) for c in self._sty_cols], dtype=np.float32)
        sty_std = (sty_vec - self._sty_mean) / self._sty_std
        sty_std = np.nan_to_num(sty_std, nan=0.0)

        berturk_cfg = self._emb_cfg["models"]["berturk"]
        emb = embed_berturk(
            [cleaned],
            model_id=berturk_cfg["model_id"],
            pooling=self._emb_cfg.get("pooling", "cls"),
            max_length=self._emb_cfg.get("max_length", 512),
            batch_size=1,
            device=self._emb_cfg.get("device", "auto"),
            cache_dir=None,
        )[0]
        emb_std = (emb - self._emb_mean) / self._emb_std

        x = np.concatenate([sty_std, emb_std]).astype(np.float32).reshape(1, -1)
        proba = self._model.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self._label_names[pred_idx]

        return PredictionResult(
            label=pred_label,
            label_tr=self._label_tr[pred_label],
            probabilities={self._label_names[i]: float(proba[i]) for i in range(len(self._label_names))},
            token_count=token_count,
            sentence_count=sentence_count,
            perplexity=perplexity,
            burstiness=burstiness,
            raw_features=feats,
            warnings=warnings,
        )


_ENGINES: dict[str, InferenceEngine] = {}


def get_engine(variant: str = "production") -> InferenceEngine:
    """variant='production' -> 3-sinifli nihai model, 'binary' -> deneysel insan/AI ikili model.

    Her iki varyant da AYNI arayuz/artifacts/ (scaler, referans, length_residualizer) dosyalarini
    kullanir -- ozellik cikarim pipeline'i degismiyor, sadece nihai siniflandirici ve etiket
    kumesi farkli.
    """
    if variant not in _ENGINES:
        if variant == "binary":
            _ENGINES[variant] = InferenceEngine(
                model_file=BINARY_MODEL_FILE,
                label_names=BINARY_LABEL_NAMES,
                label_tr=BINARY_LABEL_TR,
            )
        else:
            _ENGINES[variant] = InferenceEngine()
    return _ENGINES[variant]
