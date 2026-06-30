# Humanized AI Text Detection (TR)

Türkçe metinlerde insan / ham-AI / insanlaştırılmış-AI sınıflandırması için stilometri + transformer tabanlı üç sınıflı model.

## Kurulum

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
```

## Pipeline

| Script | Aşama |
|---|---|
| `scripts/collect_data.py` | Veri toplama (human / ai_raw / ai_humanized) |
| `scripts/preprocess.py` | Ön işleme (temizleme, tokenizasyon, POS, perplexity, burstiness) |
| `scripts/build_features.py` | Stilometrik özellikler + embedding + early-fusion |
| `scripts/train_model.py` | Model eğitimi (XGBoost / CatBoost / MLP / LogReg + stacking) |
| `scripts/tune_hpo.py` | Optuna ile hiperparametre optimizasyonu |
| `scripts/evaluate.py` | Değerlendirme (F1, ROC-AUC, confusion matrix, UMAP) |
| `scripts/explain.py` | SHAP açıklanabilirlik analizi |
| `scripts/export_secondary.py` | İkincil çıktılar (LLM-distance, anomaly heatmap) |
