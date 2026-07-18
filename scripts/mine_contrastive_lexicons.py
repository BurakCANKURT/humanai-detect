"""ai_raw <-> human contrastive n-gram lexicon madenciligi.

Girdi : data/interim/{ai_raw,human}/{label}.jsonl  (ProcessedSample'lar, tokens hazir)
Cikti :
  data/processed/lexicons/ai_cliche_ngrams.json        — ai_raw'da asiri temsil edilen n-gramlar
  data/processed/lexicons/human_informality_ngrams.json — insanda asiri temsil edilen n-gramlar

Yontem: bigram+trigram sayimlari her sinif icin cikarilir, Laplace-smoothed log-odds skoru
hesaplanir (Monroe et al. "fightin' words" yaklasiminin basitlestirilmis hali):

    score(g) = log((count_ai[g] + alpha) / (total_ai + alpha*V))
             - log((count_human[g] + alpha) / (total_human + alpha*V))

V = iki sinifin n-gram kelime-hazinesinin birlesimi. Pozitif skor -> ai_raw'a ozgu,
negatif skor -> insana ozgu. Her yonden en yuksek |skor|'lu top-K n-gram ayri bir
lexicon dosyasina yazilir (features/lexicon.py::ngram_density tarafindan tuketilir).

Sadece TRAIN split'inden madencilik yapilir: data/processed/holdout_ids.txt varsa,
o dosyadaki sample_id'ler haric tutulur (build_features.py:254-261 ile ayni kural,
holdout'a sizinti onlenir).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter

from humanai_detect.config import PROJECT_ROOT, load_yaml
from humanai_detect.preprocessing.schemas import ProcessedSample
from humanai_detect.utils.io import read_jsonl

_ORDERS = (2, 3)
_ALPHA = 0.5
_TOP_K = 300

# Tek basina virgul/nokta/tire/iki-nokta/noktali-virgul olan tokenlar -- satir-sonu
# tire-birlestirme kalintilari, tablo/atif fragmanlari (bkz. _is_noise_ngram).
_NOISE_TOKEN_RE = re.compile(r"^[,.\-:;]+$")


def _is_noise_ngram(gram: str) -> bool:
    """PDF-cikarim artefaktlarini (cid font-kodlari, kirik encoding, satir-sonu
    tire/noktalama parcalari) n-gram havuzundan eler.

    Markdown-imza karakterleri (#, *) BILEREK HARIC tutulur -- bunlar ai_raw'in gercek
    uretim davranisini yansitiyor (orn. ai_cliche listesindeki '. ####', ': **'), veri-
    toplama gurultusu degil; sadece PDF-kaynakli insan-korpusu artefaktlari hedeflenir.
    """
    if "cid" in gram.lower():
        return True
    if "�" in gram:  # Unicode replacement char -- kirik encoding
        return True
    return any(_NOISE_TOKEN_RE.match(t) for t in gram.split(" "))


def _load_samples(interim_dir, label: str, holdout_ids: set[str]) -> list[ProcessedSample]:
    path = interim_dir / label / f"{label}.jsonl"
    if not path.exists():
        return []
    samples = [ProcessedSample(**r) for r in read_jsonl(path)]
    if holdout_ids:
        samples = [s for s in samples if s.id not in holdout_ids]
    return samples


def _ngram_counts(samples: list[ProcessedSample]) -> Counter:
    counts: Counter = Counter()
    for s in samples:
        for n in _ORDERS:
            if len(s.tokens) < n:
                continue
            for i in range(len(s.tokens) - n + 1):
                gram = " ".join(s.tokens[i : i + n])
                if _is_noise_ngram(gram):
                    continue
                counts[gram] += 1
    return counts


def _log_odds_scores(counts_a: Counter, counts_b: Counter, alpha: float) -> dict[str, float]:
    vocab = set(counts_a) | set(counts_b)
    v = len(vocab)
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())

    scores = {}
    for g in vocab:
        p_a = (counts_a.get(g, 0) + alpha) / (total_a + alpha * v)
        p_b = (counts_b.get(g, 0) + alpha) / (total_b + alpha * v)
        scores[g] = math.log(p_a) - math.log(p_b)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=_TOP_K, help="Her yon icin lexicon boyutu")
    parser.add_argument("--input-dir", default=None, help="data/interim dizini (varsayilan: configs/paths.yaml)")
    parser.add_argument("--output-dir", default=None, help="lexicon ciktisinin yazilacagi dizin (varsayilan: configs/paths.yaml lexicon_dir)")
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    interim_dir = PROJECT_ROOT / (args.input_dir or paths_cfg["interim_dir"])
    processed_dir = PROJECT_ROOT / paths_cfg["processed_dir"]
    output_dir = PROJECT_ROOT / (args.output_dir or paths_cfg["lexicon_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    holdout_path = processed_dir / "holdout_ids.txt"
    holdout_ids: set[str] = set()
    if holdout_path.exists():
        holdout_ids = set(holdout_path.read_text(encoding="utf-8").splitlines())
        print(f"[mine_lexicons] holdout_ids.txt bulundu, {len(holdout_ids)} ornek madencilikten haric tutulacak.")
    else:
        print("[mine_lexicons] holdout_ids.txt yok, TUM veriden madencilik yapiliyor (henuz holdout ayrilmamis).")

    ai_samples = _load_samples(interim_dir, "ai_raw", holdout_ids)
    human_samples = _load_samples(interim_dir, "human", holdout_ids)
    print(f"[mine_lexicons] ai_raw: {len(ai_samples)} ornek, human: {len(human_samples)} ornek (train-only)")

    counts_ai = _ngram_counts(ai_samples)
    counts_human = _ngram_counts(human_samples)
    print(f"[mine_lexicons] benzersiz n-gram: ai_raw={len(counts_ai)}, human={len(counts_human)}")

    scores = _log_odds_scores(counts_ai, counts_human, _ALPHA)
    ranked = sorted(scores.items(), key=lambda kv: kv[1])

    human_top = ranked[: args.top_k]                 # en negatif skor -> insana ozgu
    ai_top = list(reversed(ranked[-args.top_k :]))    # en pozitif skor -> ai_raw'a ozgu

    ai_path = output_dir / "ai_cliche_ngrams.json"
    human_path = output_dir / "human_informality_ngrams.json"
    ai_path.write_text(json.dumps(dict(ai_top), ensure_ascii=False, indent=2), encoding="utf-8")
    human_path.write_text(json.dumps(dict(human_top), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[mine_lexicons] ai_cliche_ngrams ({len(ai_top)}) -> {ai_path}")
    print(f"  ornek: {[g for g, _ in ai_top[:5]]}")
    print(f"[mine_lexicons] human_informality_ngrams ({len(human_top)}) -> {human_path}")
    print(f"  ornek: {[g for g, _ in human_top[:5]]}")


if __name__ == "__main__":
    main()
