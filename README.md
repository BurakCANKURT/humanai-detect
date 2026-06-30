# Humanized AI Text Detection (TR)

İnsanlaştırılmış (Humanized) Yapay Metinlerin Tespiti İçin Stilometrik + Transformer Tabanlı Üç Sınıflı Model — TÜBİTAK 2209-A araştırma önerisi (`2209_A_arastirma_onerisi_formu_FINAL.pdf`) kapsamında geliştirilen kod tabanı.

Bu, henüz bir **iskelet** kuruluma karşılık gelir: modül/dosya organizasyonu ve fonksiyon imzaları hazır, asıl algoritmaların gövdesi doldurulmamıştır (`TODO` / `NotImplementedError`).

## Kurulum

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env   # ve doldur
```

## Pipeline Aşamaları ve Script Karşılıkları

| Script | Aşama |
|---|---|
| `scripts/collect_data.py` | 1. Veri toplama (insan / ham-AI / humanized) |
| `scripts/preprocess.py` | 2. Ön işleme (temizleme, tokenizasyon, POS, dependency, perplexity, burstiness) |
| `scripts/build_features.py` | 3+4+5. Stilometrik özellikler + embedding + early-fusion |
| `scripts/train_model.py` | 6. Model eğitimi (XGBoost/CatBoost/MLP/LogReg + stacking) |
| `scripts/tune_hpo.py` | 6. Optuna ile hiperparametre optimizasyonu |
| `scripts/evaluate.py` | 7. Değerlendirme (confusion matrix, F1, ROC-AUC, UMAP) |
| `scripts/explain.py` | 8. SHAP açıklanabilirlik analizi |
| `scripts/export_secondary.py` | 9. İkincil çıktılar (LLM-distance, anomaly heatmap vb.) |

## Klasör Yapısı

- `configs/` — YAML konfigürasyonları (veri kaynakları, özellik bayrakları, model hiperparametreleri)
- `src/humanai_detect/` — paket kodu (`pip install -e .` ile editable kurulur)
- `data/` — `raw/` → `interim/` → `processed/` (git'e dahil edilmez)
- `outputs/` — `models/`, `figures/`, `reports/` (git'e dahil edilmez)
- `tests/` — pytest birim/entegrasyon testleri
