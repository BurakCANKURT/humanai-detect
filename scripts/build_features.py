"""Asama 3+4: stilometrik ozellik cikarimi + transformer embedding.

Girdi : data/interim/{label}/{label}.jsonl  (ProcessedSample'lar)
Cikti :
  data/processed/stylometric.parquet         — ~30 stilometrik ozellik + id + label
  data/processed/embeddings_berturk.parquet  — [N, 768] BERTurk vektorleri
  data/processed/embeddings_roberta.parquet  — [N, 768] RoBERTa-TR vektorleri

Her ornek icin features.aggregator.extract_all_features() calisir.
Referans istatistikleri (KL div, depth shift, conjunction deviation) insan
egitim kumesinden hesaplanip diger siniflar icin yeniden kullanilir.
Embedding ciktisi --skip-embeddings bayragi ile atlanabilir (Stanza modeli yoksa).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.features.aggregator import extract_all_features
from humanai_detect.fusion.length_residualize import (
    fit_length_residualizer, apply_length_residualizer_df,
)
from humanai_detect.preprocessing.schemas import ProcessedSample
from humanai_detect.utils.io import read_jsonl, write_parquet

LABELS = ["human", "ai_raw", "ai_humanized"]


def _load_samples(interim_dir, label: str) -> list[ProcessedSample]:
    path = interim_dir / label / f"{label}.jsonl"
    if not path.exists():
        return []
    return [ProcessedSample(**r) for r in read_jsonl(path)]


class _UnionFind:
    """Ayni kaynaktan (veya birebir ayni metinden) gelen gruplari birlestirmek icin basit DSU."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _build_groups(all_samples: list[ProcessedSample]) -> pd.DataFrame:
    """Her ornek icin CV-grouping amacli bir kaynak grubu (group_id) turetir.

    Amac: StratifiedGroupKFold ile ayni kaynak dokumandan/prompt'tan gelen orneklerin
    train/validation arasinda bolunmesini (leakage) onlemek.
      - human       : DergiPark ise oai_id, degilse kaynak dosya adi (filename) -- back-translate
                      edilmis insan augmentasyonu (metadata.original_id) kaynak belgenin grubunu
                      miras alir (bkz. scripts/humanize_human_topup.py, 2026-07-21).
      - ai_raw      : uretimde kullanilan prompt metni (sinirli sayida sablon var).
      - ai_humanized: eslesen ai_raw orneginin (metadata.original_id) grubunu miras alir.

    Ayrica: metadata farkli olsa bile (orn. DergiPark'ta ayni makalenin iki farkli oai_id
    altinda mukerrer indekslenmesi) BIREBIR AYNI temiz metne sahip orneklerin gruplari
    Union-Find ile birlestirilir — aksi halde bu kopyalar train/validation arasina
    bolunup dogrudan (seyreltilmemis) veri sizintisina yol acabilir.
    """
    ai_raw_topic_by_id: dict[str, str] = {}
    prompt_to_topic: dict[str, str] = {}
    for s in all_samples:
        if s.label != "ai_raw":
            continue
        prompt = (s.metadata or {}).get("prompt", "")
        topic = prompt_to_topic.setdefault(prompt, f"ai_topic_{len(prompt_to_topic):02d}")
        ai_raw_topic_by_id[s.id] = topic

    # Insan augmentasyonu (back-translate edilmis "human" ornekleri, bkz.
    # scripts/humanize_human_topup.py) kaynak belgenin grubunu miras alabilsin diye,
    # ONCE gercek (augmentasyon OLMAYAN) insan belgelerinin gruplarini hesapla.
    human_group_by_id: dict[str, str] = {}
    for s in all_samples:
        if s.label != "human":
            continue
        md = s.metadata or {}
        if "original_id" in md:
            continue  # bu zaten bir augmentasyon, asagida ikinci geciste cozulecek
        if "oai_id" in md:
            human_group_by_id[s.id] = f"human_doc_oai_{md['oai_id']}"
        elif "filename" in md:
            human_group_by_id[s.id] = f"human_doc_file_{md['filename']}"
        else:
            human_group_by_id[s.id] = f"human_doc_{s.id}"

    prelim: dict[str, str] = {}
    for s in all_samples:
        md = s.metadata or {}
        if s.label == "human":
            orig_id = md.get("original_id")
            if orig_id is not None and orig_id in human_group_by_id:
                group_id = human_group_by_id[orig_id]
            elif s.id in human_group_by_id:
                group_id = human_group_by_id[s.id]
            else:
                group_id = f"human_doc_{s.id}"  # tekil (kaynak bilgisi yok)
        elif s.label == "ai_raw":
            group_id = ai_raw_topic_by_id.get(s.id, f"ai_topic_unknown_{s.id}")
        elif s.label == "ai_humanized":
            orig_id = md.get("original_id")
            group_id = ai_raw_topic_by_id.get(orig_id, f"ai_topic_unknown_{s.id}")
        else:
            group_id = s.id
        prelim[s.id] = group_id

    # Birebir ayni temiz metne sahip orneklerin (farkli kaynak/oai_id'ye ragmen)
    # gruplarini birlestir.
    uf = _UnionFind()
    text_first_group: dict[str, str] = {}
    for s in all_samples:
        text = s.cleaned_text.strip()
        gid = prelim[s.id]
        if text in text_first_group:
            uf.union(gid, text_first_group[text])
        else:
            text_first_group[text] = gid

    rows = [
        {"sample_id": s.id, "label": s.label, "group_id": uf.find(prelim[s.id])}
        for s in all_samples
    ]
    return pd.DataFrame(rows)


def _build_reference(human_samples: list[ProcessedSample]) -> dict:
    """Insan egitim kumesinden referans istatistiklerini hesaplar."""
    if not human_samples:
        return {}
    all_tokens: list[str] = []
    dep_depths: list[float] = []
    conj_densities: list[float] = []

    for s in human_samples:
        all_tokens.extend(s.tokens)
        from humanai_detect.features.syntactic import _word_depths
        from humanai_detect.features.discourse import conjunction_density

        if s.dep_parse:
            dep_depths.extend(_word_depths(s.dep_parse))
        conj_densities.append(conjunction_density(s.tokens, s.pos_tags))

    import statistics
    total = len(all_tokens)
    counts = Counter(all_tokens)
    return {
        "word_freqs": {w: c / total for w, c in counts.items()},
        "mean_dep_depth": statistics.mean(dep_depths) if dep_depths else 0.0,
        "conjunction_density": statistics.mean(conj_densities) if conj_densities else 0.0,
    }


def _load_lexicons(processed_dir, paths_cfg: dict) -> dict:
    """scripts/mine_contrastive_lexicons.py ciktisi lexicon JSON'larini yukler.

    Dosyalar yoksa (henuz madencilik yapilmamissa) bos kumeler doner -- ilgili
    ozellikler aggregator.py'de 0.0 olarak hesaplanir, pipeline durmaz.
    """
    lexicon_dir = PROJECT_ROOT / paths_cfg.get("lexicon_dir", "data/processed/lexicons")
    lexicons = {"ai_cliche": set(), "human_informality": set()}

    ai_path = lexicon_dir / "ai_cliche_ngrams.json"
    human_path = lexicon_dir / "human_informality_ngrams.json"
    if ai_path.exists():
        lexicons["ai_cliche"] = set(json.loads(ai_path.read_text(encoding="utf-8")).keys())
    if human_path.exists():
        lexicons["human_informality"] = set(json.loads(human_path.read_text(encoding="utf-8")).keys())

    print(f"[lexicons] ai_cliche={len(lexicons['ai_cliche'])}, human_informality={len(lexicons['human_informality'])}"
          + ("" if ai_path.exists() and human_path.exists() else " (scripts/mine_contrastive_lexicons.py henuz calistirilmamis)"))
    return lexicons


def _embed_all(
    samples: list[ProcessedSample],
    model_key: str,
    emb_cfg: dict,
    paths_cfg: dict,
    processed_dir,
) -> None:
    """Tum ornekler icin embedding hesaplar ve parquet olarak kaydeder."""
    from humanai_detect.embeddings.berturk import embed_berturk
    from humanai_detect.embeddings.roberta_tr import embed_roberta_tr

    model_cfg = emb_cfg["models"][model_key]
    if not model_cfg.get("enabled", True):
        return

    model_id = model_cfg["model_id"]
    pooling = emb_cfg.get("pooling", "cls")
    max_length = emb_cfg.get("max_length", 512)
    batch_size = emb_cfg.get("batch_size", 16)
    device = emb_cfg.get("device", "auto")

    cache_dir = None
    if emb_cfg.get("cache_enabled"):
        # DUZELTME (2026-07-19): emb_cfg["cache_dir"] configs/embeddings.yaml'da ZATEN
        # tam yol ("data/processed/embedding_cache") -- processed_dir'i tekrar eklemek
        # cache'i hic kullanilmayan cift-katli bir klasore ("data/processed/data/processed/...")
        # yoniendiriyordu, bkz. proje notlari.
        cache_dir = PROJECT_ROOT / emb_cfg.get("cache_dir", "data/processed/embedding_cache") / model_key

    embed_fn = embed_berturk if model_key == "berturk" else embed_roberta_tr
    out_path = processed_dir / f"embeddings_{model_key}.parquet"

    if out_path.exists():
        print(f"[{model_key}] {out_path} zaten mevcut, atlanıyor.")
        return

    print(f"[{model_key}] {len(samples)} metin icin embedding hesaplaniyor ({model_id})...")
    texts = [s.cleaned_text for s in samples]
    emb_matrix = embed_fn(
        texts, model_id=model_id, pooling=pooling,
        max_length=max_length, batch_size=batch_size,
        device=device, cache_dir=cache_dir,
    )

    dim_cols = {f"dim_{i}": emb_matrix[:, i] for i in range(emb_matrix.shape[1])}
    df_emb = pd.DataFrame({
        "sample_id": [s.id for s in samples],
        "label": [s.label for s in samples],
        **dim_cols,
    })
    write_parquet(df_emb, out_path)
    print(f"[{model_key}] {len(df_emb)} ornek, {emb_matrix.shape[1]} boyut -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Embedding adimini atla (sadece stilometrik ozellikler)")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    features_cfg = load_yaml("features")
    emb_cfg = load_yaml("embeddings")

    interim_dir = PROJECT_ROOT / (args.input_dir or paths_cfg["interim_dir"])
    processed_dir = PROJECT_ROOT / (args.output_dir or paths_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Referans istatistiklerini insan kumesinden hesapla
    print("[build_features] insan kumesi referans istatistikleri hesaplaniyor...")
    human_samples = _load_samples(interim_dir, "human")
    reference = _build_reference(human_samples)
    print(f"  referans hazir ({len(human_samples)} insan ornegi)")

    lexicons = _load_lexicons(processed_dir, paths_cfg)

    # --- Asama 3: Stilometrik ozellikler ---
    all_samples: list[ProcessedSample] = []
    rows: list[dict] = []
    token_counts: list[int] = []
    for label in LABELS:
        samples = _load_samples(interim_dir, label)
        if not samples:
            print(f"[{label}] veri yok, atlanıyor.")
            continue
        all_samples.extend(samples)
        print(f"[{label}] {len(samples)} ornek icin ozellik cikarimi basliyor...")
        for i, sample in enumerate(samples, 1):
            feats = extract_all_features(sample, features_cfg, reference=reference, lexicons=lexicons)
            feats["sample_id"] = sample.id
            feats["label"] = sample.label
            rows.append(feats)
            token_counts.append(sample.token_count)
            if i % 10 == 0 or i == len(samples):
                print(f"  [{i}/{len(samples)}] {label}", flush=True)

    if not rows:
        print("[build_features] hicbir ornek islenmedi.")
        return

    df = pd.DataFrame(rows)
    cols = ["sample_id", "label"] + [c for c in df.columns if c not in ("sample_id", "label")]
    df = df[cols]

    # --- Uzunluk-confound residualizasyonu (SHAP top-4 ozellik) ---
    # train_mask: held-out set zaten belirlenmisse (holdout_ids.txt), residualizasyon
    # regresyonu SADECE dev havuzundan fit edilir (sizinti onleme, standardize.py'deki
    # train_mask deseniyle ayni mantik).
    token_counts_arr = np.array(token_counts, dtype=float)
    holdout_path = processed_dir / "holdout_ids.txt"
    train_mask = None
    if holdout_path.exists():
        holdout_ids = set(holdout_path.read_text(encoding="utf-8").splitlines())
        train_mask = ~df["sample_id"].isin(holdout_ids).to_numpy()
        print(f"[length-residualize] train_mask: {train_mask.sum()}/{len(df)} ornek (held-out haric fit)")
    else:
        print("[length-residualize] holdout_ids.txt yok, TUM veriyle fit ediliyor (henuz holdout ayrilmamis).")

    length_params = fit_length_residualizer(df, token_counts_arr, train_mask=train_mask)
    df = apply_length_residualizer_df(df, token_counts_arr, length_params)
    params_path = processed_dir / "length_residualizer.json"
    params_path.write_text(json.dumps(length_params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[length-residualize] {list(length_params.keys())} -> {params_path}")

    out_path = processed_dir / "stylometric.parquet"
    write_parquet(df, out_path)
    print(f"[stilometri] {len(df)} ornek, {len(df.columns)-2} ozellik -> {out_path}")

    # --- Grouping (CV leakage onleme) ---
    groups_df = _build_groups(all_samples)
    groups_path = processed_dir / "groups.parquet"
    write_parquet(groups_df, groups_path)
    n_groups = groups_df["group_id"].nunique()
    print(f"[groups] {len(groups_df)} ornek, {n_groups} benzersiz kaynak grubu -> {groups_path}")

    # --- Asama 4: Embedding ---
    if not args.skip_embeddings and all_samples:
        for model_key in emb_cfg.get("models", {}):
            _embed_all(all_samples, model_key, emb_cfg, paths_cfg, processed_dir)

    # --- Asama 5: Early Fusion ---
    fused_path = processed_dir / "fused.parquet"
    sty_path = processed_dir / "stylometric.parquet"
    if not sty_path.exists():
        print("[fusion] stylometric.parquet bulunamadi, fusion atlandi.")
        return

    from humanai_detect.fusion.early_fusion import build_fused_dataframe
    from humanai_detect.utils.io import read_parquet

    fusion_cfg = load_yaml("fusion")
    sty_df = pd.read_parquet(sty_path)

    emb_named: list[tuple[str, pd.DataFrame]] = []
    for model_key in emb_cfg.get("models", {}):
        emb_path = processed_dir / f"embeddings_{model_key}.parquet"
        if emb_path.exists():
            emb_named.append((model_key, pd.read_parquet(emb_path)))

    # Scaler (standardize/PCA) da uzunluk-residualizasyonuyla ayni mantikla SADECE dev
    # havuzundan fit edilmeli. sty_df, stylometric.parquet'ten (yukaridaki df'in disk
    # round-trip'i) okundugu icin satir sirasi df ile ayni -- train_mask dogrudan gecerli.
    assert len(sty_df) == len(df), "stylometric.parquet satir sayisi df ile uyusmuyor"
    fused_df = build_fused_dataframe(sty_df, emb_named, fusion_cfg, train_mask=train_mask)
    write_parquet(fused_df, fused_path)
    n_feat = len(fused_df.columns) - 2
    print(f"[fusion] {len(fused_df)} ornek, {n_feat} toplam ozellik -> {fused_path}")


if __name__ == "__main__":
    main()
