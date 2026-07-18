"""Regex-tabanli ortografik duzensizlik ozellikleri (lexicon/model gerektirmez).

Turkce icin ozel bir yazim-denetim kutuphanesi (zeyrek/hunspell/symspell) projede mevcut
degil; bunun yerine dogrudan gozlemlenebilir yuzey duzensizliklerini (tekrarli noktalama,
cift bosluk, noktalama-sonrasi buyuk harf tutarsizligi) olcuyoruz. Insan metni bu tur
"gurultu"yu AI ciktisindan (humanize edilmis olsa bile) belirgin sekilde daha sik icerir.
"""

from __future__ import annotations

import re

_REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1+|[!?]{2,}|\.{4,}")
_DOUBLE_SPACE_RE = re.compile(r"  +")
_LOWERCASE_AFTER_END_RE = re.compile(r"[.!?]\s+[a-zçğıöşü]")
_ANY_AFTER_END_RE = re.compile(r"[.!?]\s+[a-zA-ZçğıöşüÇĞİÖŞÜ]")


def punct_irregularity_rate(text: str) -> float:
    """Tekrarli noktalama (!!, ??, ...., vb.) yogunlugu, karakter basina."""
    if not text:
        return 0.0
    matches = len(_REPEATED_PUNCT_RE.findall(text))
    return matches / len(text)


def double_space_rate(text: str) -> float:
    """Cift/coklu bosluk yogunlugu, kelime basina."""
    words = text.split()
    if not words:
        return 0.0
    matches = len(_DOUBLE_SPACE_RE.findall(text))
    return matches / len(words)


def post_punct_case_irregularity_rate(text: str) -> float:
    """Cumle-sonu noktalamadan sonra kucuk harfle baslama orani (buyuk harf beklenirken).

    Turkce yazim kuralina gore cumle basi buyuk harfle baslamalidir; bu kuralin ne siklikta
    ihlal edildigini, tum noktalama-sonrasi-harf gecislerine oranla olcer.
    """
    total = len(_ANY_AFTER_END_RE.findall(text))
    if total == 0:
        return 0.0
    irregular = len(_LOWERCASE_AFTER_END_RE.findall(text))
    return irregular / total
