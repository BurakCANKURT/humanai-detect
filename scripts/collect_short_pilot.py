"""Kisa-metin pilot verisi toplama (5-30 kelime araligi).

Amac: Ana veri seti min_tokens=30 sinirinin ALTINDA hicbir ornek icermiyor (bkz.
configs/preprocessing.yaml). Model bu yuzden gercekten kisa metinlerde (kullanicilarin
gercek dunyada girebilecegi 10-30 kelimelik paragraflar gibi) hic egitim gormemis
bir bolgeye dusuyor ve OOD/asiri-guvenli davraniyor (her zaman "human" tahmini,
[[project-codebase]] 2026-07-16 notu). Bu script, ana 3000/sinif veri setine
DOKUNMADAN, ayri bir kucuk (varsayilan 300/sinif) "short pilot" havuzu uretir:

  human        : mevcut data/raw/human/human.jsonl kayitlarini (zaten indirilmis/
                 cikarilmis metin) 5-30 kelimelik parcalara YENIDEN boler. Yeni
                 indirme/API gerekmez, tamamen yerel CPU'da calisir, Colab GEREKMEZ.
  ai_raw       : Qwen2.5-7B-Instruct (transformers) ile OZELLIKLE KISA (5-30 kelime)
                 hedef uzunlukla YENI metin uretir. GPU gerekir (Colab).
  ai_humanized : uretilen kisa ai_raw'i mevcut back-translation humanizer'i (kucuk MT
                 modeli) ile "insansilastirir". CPU'da calisir ama Colab'da GPU varken
                 daha hizlidir; ai_raw ile ayni Colab oturumunda calistirilmasi onerilir.

Cikti: data/raw/{human,ai_raw,ai_humanized}_short/{label}_short.jsonl
       (ana data/raw/{label}/{label}.jsonl dosyalarindan TAMAMEN AYRI, uzerine yazmaz)

Kullanim (sirayla, --label ai_raw ve ai_humanized Colab GPU gerektirir):
    python scripts/collect_short_pilot.py --label human
    python scripts/collect_short_pilot.py --label ai_raw
    python scripts/collect_short_pilot.py --label ai_humanized

Checkpoint/resume: ana collect_data.py ile ayni mantik -- yarida kesilirse ayni
komutla devam edilebilir, mevcut kayitlar tekrar uretilmez/silinmez.
"""
from __future__ import annotations

import argparse
import random
import re
from dataclasses import asdict

from humanai_detect.config import PROJECT_ROOT, get_api_key, load_yaml
from humanai_detect.data_collection import humanizers, llm_generators
from humanai_detect.data_collection.file_ingest import chunk_text
from humanai_detect.data_collection.schemas import RawSample
from humanai_detect.utils.io import read_jsonl, write_jsonl

# dergipark_harvest'in _bibliography_ratio/_is_turkish fonksiyonlari SADECE
# collect_human_short() icinde, fonksiyon ici (lazy) import edilir -- bu modul
# fitz (PyMuPDF) ve py3langid'e bagimli; ai_raw/ai_humanized (Colab, GPU) yollari
# bunlara hic ihtiyac duymuyor, gereksiz bagimlilik/kurulum sartina yol acmasin.

PROVIDER_TAG = "transformers_short"

# Belge basindaki dergi/makale basligi, yazar adi/unvani, footnote, "kaynak goster"
# ifadesi gibi duz anlatim OLMAYAN metinleri elemek icin sezgisel filtre. Ana DergiPark
# harvester'indaki _bibliography_ratio (yil-parantez/pp./ss. kaliplari) ile ayni
# motivasyona sahiptir (bkz. dergipark_harvest.py): baslik/kaynakca gibi yapay ipuclari,
# modelin gercek uslup farkindan degil bu tesadufi kaliplardan ogrenmesine yol acabilir.
# Kisa (5-30 kelime) chunk'larda TEK bir kaynakca/footnote satiri bile chunk'in tamamini
# kirletebildigi icin (uzun chunk'lardaki gibi bir "oran" esigi degil) esik cok dusuk tutulur.
_HEADER_KEYWORDS_RE = re.compile(
    r"\b(cilt|say[ıi]|issn|vol\.|pp\.|ss\.|s\.\s*\d|dergisi|abstract|özet/abstract|"
    r"öğr\.?\s*gör\.?|doç\.?\s*dr\.?|prof\.?\s*dr\.?|to cite this article|curr res)\b",
    re.IGNORECASE,
)
_PAGE_RANGE_RE = re.compile(r"^\s*\d+[-–]\d+\s*$")
_FOOTNOTE_PREFIX_RE = re.compile(r"^\s*[*\d]+\.?\s*[A-ZÇĞİÖŞÜ]")
# Ard arda birden fazla "12 Yazar," turu footnote/kaynakca satiri -- tek basina bir
# rakam+ozel-isim rastlantisal olabilir ama 2+ tekrari neredeyse her zaman kaynakca listesi.
_MULTI_FOOTNOTE_RE = re.compile(r"\d+\s+[A-ZÇĞİÖŞÜ][a-zçğıöşüi]+,")
# "(3), 53-66" / "16(3): 568-574" gibi cilt(sayi), sayfa-araligi kaynakca kalibi.
_CITATION_PAGES_RE = re.compile(r"\(\d+\)[,:]?\s*\d+[-–]\d+")


def _looks_like_header(text: str) -> bool:
    # Lazy import: bu iki fonksiyon dergipark_harvest.py'de tanimli, fitz (PyMuPDF)
    # ve py3langid'e bagimli bir modulden geliyor -- sadece human_short yolunda
    # gerekli, ai_raw/ai_humanized (Colab, GPU) calistirmalarinda hic ihtiyac
    # duyulmuyor, modul-seviyesi import olsaydi onlar icin de gereksiz kurulum
    # sartina yol acardi.
    from humanai_detect.data_collection.dergipark_harvest import _bibliography_ratio, _is_turkish

    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    if upper_ratio > 0.4:
        return True
    if _HEADER_KEYWORDS_RE.search(text):
        return True
    if _PAGE_RANGE_RE.match(text):
        return True
    if _FOOTNOTE_PREFIX_RE.match(text):
        return True
    if len(_MULTI_FOOTNOTE_RE.findall(text)) >= 2:
        return True
    if _CITATION_PAGES_RE.search(text):
        return True
    if _bibliography_ratio(text) > 0:
        return True
    # DergiPark makaleleri bazen Ingilizce ozet/kaynak cumleleri de icerir -- ana
    # veri setinde yok_tez kaynaginin tamami Ingilizce ciktigi icin tamamen elenmisti
    # (bkz. proje notlari, filter_human_language.py); ayni standardi burada da uyguluyoruz.
    if len(text.split()) >= 6 and not _is_turkish(text):
        return True
    return False


def collect_human_short(paths_cfg: dict, data_sources_cfg: dict) -> list[RawSample]:
    """Mevcut human.jsonl kayitlarini kisa (5-30 kelime) parcalara yeniden boler.

    Her belgeden birden fazla (max _MAX_CHUNKS_PER_DOC) parca toplanip TUM havuzdan
    rastgele (seed=42) secim yapilir -- boylece her belgenin sadece ILK cumlesini
    (genelde dergi basligi/yazar adi gibi duz-anlatim-olmayan bir header) almak yerine
    govde metninden temsili bir ornek elde edilir. Header benzeri parcalar (_looks_like_header)
    ve max_words'u asan parcalar (chunk_text tek bir uzun cumleyi oldugu gibi birakabilir)
    tamamen elenir.
    """
    sp_cfg = data_sources_cfg["short_pilot"]
    target_count = sp_cfg["target_count_per_class"]
    min_words = sp_cfg["human_min_words"]
    max_words = sp_cfg["human_max_words"]
    max_chunks_per_doc = 3

    src_path = PROJECT_ROOT / paths_cfg["raw_dir"] / "human" / "human.jsonl"
    if not src_path.exists():
        raise FileNotFoundError(
            f"{src_path} bulunamadi. Ana human verisi once toplanmis olmali "
            "(bkz. collect_data.py --label human)."
        )
    source_records = [RawSample(**r) for r in read_jsonl(src_path)]

    candidates: list[tuple[str, str, str]] = []  # (text, original_id, original_source)
    for record in source_records:
        pieces = chunk_text(record.text, min_words=min_words, max_words=max_words)
        kept = 0
        for piece in pieces:
            if kept >= max_chunks_per_doc:
                break
            word_count = len(piece.split())
            if not (min_words <= word_count <= max_words):
                continue
            if _looks_like_header(piece):
                continue
            candidates.append((piece, record.id, record.source))
            kept += 1

    rng = random.Random(42)
    rng.shuffle(candidates)
    selected = candidates[:target_count]

    samples = [
        RawSample(
            id=f"human_short_{idx:04d}",
            text=text,
            label="human",
            source="human_short_rechunk",
            metadata={"original_id": orig_id, "original_source": orig_source},
        )
        for idx, (text, orig_id, orig_source) in enumerate(selected)
    ]

    print(f"[human_short] {len(source_records)} kaynak kayittan {len(candidates)} aday parca "
          f"(header filtrelenmis) uretildi, {len(samples)} tanesi secildi "
          f"({min_words}-{max_words} kelime).")
    return samples


def collect_ai_raw_short(paths_cfg: dict, data_sources_cfg: dict) -> list[RawSample]:
    """Qwen ile ozellikle kisa (5-30 kelime) hedefli YENI ai_raw metni uretir."""
    sp_cfg = data_sources_cfg["short_pilot"]
    target_count = sp_cfg["target_count_per_class"]
    out_path = PROJECT_ROOT / paths_cfg["raw_dir"] / "ai_raw_short" / "ai_raw_short.jsonl"

    existing: list[RawSample] = []
    if out_path.exists():
        existing = [RawSample(**r) for r in read_jsonl(out_path)]
        if len(existing) >= target_count:
            print(f"[ai_raw_short] {len(existing)} ornek zaten mevcut, atlanıyor.")
            return existing[:target_count]
        print(f"[ai_raw_short] {len(existing)} mevcut ornek yuklendi, "
              f"{target_count - len(existing)} daha uretilecek.")

    transformers_cfg = data_sources_cfg["llm_generators"]["transformers"]
    prompts = llm_generators.load_prompts(
        PROJECT_ROOT / data_sources_cfg["llm_generators"]["prompts_file"]
    )
    remaining = target_count - len(existing)

    new_samples = llm_generators.generate_batch(
        prompts,
        PROVIDER_TAG,
        remaining,
        checkpoint_path=out_path,
        start_index=len(existing),
        model_id=transformers_cfg["model"],
        device=transformers_cfg.get("device", "auto"),
        load_in_4bit=transformers_cfg.get("load_in_4bit", True),
        batch_size=transformers_cfg.get("batch_size", 8),
        target_len_mean=sp_cfg["target_len_mean"],
        target_len_std=sp_cfg["target_len_std"],
        target_len_min=sp_cfg["target_len_min"],
        target_len_max=sp_cfg["target_len_max"],
        max_new_tokens=sp_cfg["max_new_tokens"],
    )
    return (existing + new_samples)[:target_count]


def collect_ai_humanized_short(paths_cfg: dict, data_sources_cfg: dict) -> list[RawSample]:
    """Kisa ai_raw ornaklerini back-translation ile humanize eder."""
    ai_raw_short_path = PROJECT_ROOT / paths_cfg["raw_dir"] / "ai_raw_short" / "ai_raw_short.jsonl"
    if not ai_raw_short_path.exists():
        raise FileNotFoundError(
            f"{ai_raw_short_path} bulunamadi. Once '--label ai_raw' ile kisa ham-AI verisini toplayin."
        )
    ai_raw_samples = [RawSample(**r) for r in read_jsonl(ai_raw_short_path)]

    out_path = PROJECT_ROOT / paths_cfg["raw_dir"] / "ai_humanized_short" / "ai_humanized_short.jsonl"
    existing: list[RawSample] = []
    if out_path.exists():
        existing = [RawSample(**r) for r in read_jsonl(out_path)]
    start_index = len(existing)
    if start_index >= len(ai_raw_samples):
        print(f"[ai_humanized_short] {start_index} ornek zaten checkpoint'te — atlanıyor.")
        return existing

    backtranslate_cfg = data_sources_cfg["humanizers"]["llm"]["backtranslate"]
    new_samples = humanizers.humanize_batch_llm(
        ai_raw_samples,
        "backtranslate",
        checkpoint_path=out_path,
        start_index=start_index,
        tr_en_model=backtranslate_cfg["tr_en_model"],
        en_tr_model=backtranslate_cfg["en_tr_model"],
        device=backtranslate_cfg.get("device", "auto"),
    )
    return existing + new_samples


def _write(label: str, samples: list[RawSample], paths_cfg: dict) -> None:
    out_dir_label = f"{label}_short"
    out_path = PROJECT_ROOT / paths_cfg["raw_dir"] / out_dir_label / f"{out_dir_label}.jsonl"
    # ai_raw_short/ai_humanized_short checkpoint ile aninda yazildi; sayim eslesirse tekrar yazma.
    if label in ("ai_raw", "ai_humanized") and out_path.exists():
        existing_count = sum(1 for _ in read_jsonl(out_path))
        if existing_count >= len(samples):
            print(f"[{out_dir_label}] {existing_count} ornek checkpoint'ten zaten yazilmis -> {out_path}")
            return
    write_jsonl(out_path, (asdict(s) for s in samples))
    print(f"[{out_dir_label}] {len(samples)} ornek yazildi -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        choices=["human", "ai_raw", "ai_humanized"],
        required=True,
        help="Hangi sinif icin kisa-pilot verisi toplanacak",
    )
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    data_sources_cfg = load_yaml("data_sources")

    collectors = {
        "human": collect_human_short,
        "ai_raw": collect_ai_raw_short,
        "ai_humanized": collect_ai_humanized_short,
    }
    samples = collectors[args.label](paths_cfg, data_sources_cfg)
    _write(args.label, samples, paths_cfg)


if __name__ == "__main__":
    main()
