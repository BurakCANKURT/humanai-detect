"""Contrastive-mining lexicon'lara dayali n-gram yogunluk ozellikleri.

scripts/mine_contrastive_lexicons.py tarafindan uretilen iki lexicon icin ayni fonksiyon
kullanilir:
  - ai_cliche_ngrams        : ai_raw korpusunda insan korpusuna kiyasla asiri temsil edilen
                               bigram/trigramlar (uretici LLM'in imza kaliplari).
  - human_informality_ngrams: insan korpusunda ai_raw'a kiyasla asiri temsil edilen
                               bigram/trigramlar (gundelik/informal kaliplar).
"""

from __future__ import annotations


def ngram_density(tokens: list[str], lexicon: set[str], orders: tuple[int, ...] = (2, 3)) -> float:
    """tokens icindeki n-gramlarin (verilen orders icin) lexicon'a dusme oranini hesaplar.

    lexicon: " " ile birlestirilmis n-gram string'lerinden olusan kume (mine_contrastive_lexicons.py ciktisi).
    """
    if not tokens or not lexicon:
        return 0.0

    total = 0
    matched = 0
    for n in orders:
        if len(tokens) < n:
            continue
        grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        total += len(grams)
        matched += sum(1 for g in grams if g in lexicon)

    return matched / total if total else 0.0
