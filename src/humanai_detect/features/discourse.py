"""Soylem duzeyi metrikler: baglac yogunlugu, fonksiyon kelime orani, cumleler-arasi tutarlilik."""

from __future__ import annotations

import math
import statistics

# Universal POS etiket kumeleri (UPOS, Stanza ciktisi)
_CONJUNCTION_POS = {"CCONJ", "SCONJ"}
_FUNCTION_WORD_POS = {"DET", "PRON", "ADP", "AUX", "PART", "SCONJ", "CCONJ", "INTJ"}

# Jaccard-tabanli tutarlilik hesabinda goz ardi edilecek kucuk bir Turkce fonksiyon-kelime
# kumesi (per-cumle POS hizalamasi mevcut olmadigi icin lexical_coherence icin ayrica tutulur).
_COHERENCE_STOPWORDS = {
    "ve", "ile", "bir", "bu", "şu", "o", "da", "de", "ki", "mi", "mı", "mu", "mü",
    "gibi", "çok", "daha", "en", "ama", "fakat", "ancak", "veya", "ya", "ise",
    "için", "olan", "olarak", "kadar", "göre", "sonra", "önce", "ne", "ben", "sen",
    "biz", "siz", "onlar", "var", "yok", "her", "hiç", "tüm", "bütün", "diğer",
    "böyle", "şöyle", "öyle", "nasıl", "neden", "niçin", "üzere",
}


def conjunction_density(tokens: list[str], pos_tags: list[tuple[str, str]]) -> float:
    """Baglac (CCONJ + SCONJ) kullanim yogunlugu: baglac sayisi / toplam token.

    AI metinleri genellikle daha az varyasyonlu, daha yuksek baglac yogunluguna sahip.
    """
    if not tokens:
        return 0.0
    conj_count = sum(1 for _, pos in pos_tags if pos in _CONJUNCTION_POS)
    return conj_count / len(tokens)


def conjunction_deviation_index(sample_density: float, reference_density: float) -> float:
    """Orneklerin baglac yogunlugunun insan referansindan bagil sapmasini olcer.

    0 -> referansla ayni; pozitif -> insandan daha baglaç-agir; negatif -> daha az.
    reference_density sifirsa 0.0 doner.
    """
    if reference_density == 0.0:
        return 0.0
    return (sample_density - reference_density) / reference_density


def function_word_ratio(tokens: list[str], pos_tags: list[tuple[str, str]]) -> float:
    """Fonksiyon kelimelerin (DET/PRON/ADP/AUX/PART/CONJ/INTJ) toplam tokenlara orani.

    AI metinleri genellikle daha yuksek fonksiyon kelime oranina sahiptir
    (daha basit ve tekrar eden yapisal kaliplar).
    """
    if not tokens:
        return 0.0
    fw_count = sum(1 for _, pos in pos_tags if pos in _FUNCTION_WORD_POS)
    return fw_count / len(tokens)


def lexical_coherence(sentences: list[str]) -> float:
    """Ardisik cumleler arasi icerik-kelime (stopword haric) Jaccard benzerliginin ortalamasi.

    Yerel soylem tutarliligi icin basit bir entity-grid yaklasimi: AI metni (humanize edilmis
    olsa bile) genelde asiri duzgun/siki yerel tutarlilik gosterir; insan metni daha
    cagrisimsal/atlamali akar. Tam entity-grid (sozdizimsel rol + coreference) yerine
    lexical-overlap kullanilir -- ProcessedSample cumle-bazli POS hizalamasi saglamiyor.

    Tek cumlelik (veya bos) metinlerde tanimsizdir -> NaN (burstiness.py ile ayni konvansiyon,
    sabit bir deger "duzenli/AI-benzeri" anlamina gelmesin diye).
    """
    if len(sentences) < 2:
        return math.nan

    word_sets = [
        {w.lower() for w in sent.split() if w.strip(".,!?;:\"'()") and w.lower() not in _COHERENCE_STOPWORDS}
        for sent in sentences
    ]

    scores = []
    for a, b in zip(word_sets, word_sets[1:]):
        union = a | b
        if not union:
            continue
        scores.append(len(a & b) / len(union))

    return statistics.mean(scores) if scores else math.nan
