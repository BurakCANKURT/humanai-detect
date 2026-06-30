"""Metin temizleme ve normalizasyon."""

from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """Unicode normalizasyonu, kontrol karakteri ve fazla bosluk temizligi yapar.

    PDF/DOCX kaynaklardan gelen gurultuyu (satir sonu hatalari, bom karakterleri,
    art arda boslukar) giderir; metnin anlamsal icerigi degistirilmez.
    """
    # NFC normalizasyonu: bilesik karakterleri kanonize et (ornek: e + ̈ → ë)
    text = unicodedata.normalize("NFC", text)

    # BOM ve null byte temizle
    text = text.replace("﻿", "").replace("\x00", "")

    # Kontrol karakterlerini kaldir (\n ve \t haric)
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Yatay boslukları (bosluk + tab) tek bosluga indir
    text = re.sub(r"[ \t]+", " ", text)

    # Uc veya daha fazla art arda satir sonunu iki satira indir
    text = re.sub(r"\n{3,}", "\n\n", text)

    # PDF'lerden gelen satir ortasi tire birlestirmelerini duzelt (kel- \nime → kelime)
    text = re.sub(r"-\n(?=[a-zA-ZğüşıöçĞÜŞİÖÇ])", "", text)

    return text.strip()
