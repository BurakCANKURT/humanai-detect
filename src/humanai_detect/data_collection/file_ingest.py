"""Elle toplanan dosyalardan (txt/pdf/docx) duz metin cikarma yardimcilari.

YOK Tez Merkezi arama sayfasi bot-dogrulama (CAPTCHA) ile korundugu icin otomatik
kazima yapilmiyor; bunun yerine kullanici dosyalari elle indirip data/external/<kaynak>/
klasorune koyar, bu modul de o klasoru okur.
"""

from __future__ import annotations

import re
from pathlib import Path

_SUPPORTED_SUFFIXES = {".txt", ".pdf", ".docx"}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def extract_text(path: Path) -> str:
    """Bir dosyadan duz metni cikarir (uzantiya gore txt/pdf/docx)."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if suffix == ".docx":
        import docx

        document = docx.Document(path)
        return "\n".join(p.text for p in document.paragraphs)
    raise ValueError(f"Desteklenmeyen dosya turu: {suffix}")


def iter_supported_files(directory: Path) -> list[Path]:
    """Bir klasordeki desteklenen (txt/pdf/docx) dosyalarini isim sirasina gore dondurur."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES)


def chunk_text(text: str, min_words: int, max_words: int) -> list[str]:
    """Uzun bir metni (orn. tam tez) cumle sinirlarinda min/max_words araligina gore parcalara boler.

    Tek bir makale/tez tek ornek yerine birden fazla, AI ciktilariyla kiyaslanabilir
    uzunlukta ornege donusur. Sinir kesin degil: bir parca min_words altinda kalirsa
    onceki parcayla birlestirilir; tek bir cumle max_words'u asarsa oldugu gibi birakilir
    (Asama 2 on isleme bu durumu ayrica filtreler/kesebilir).
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words
    if current:
        chunks.append(" ".join(current))

    if len(chunks) > 1 and len(chunks[-1].split()) < min_words:
        last = chunks.pop()
        chunks[-1] = chunks[-1] + " " + last

    return chunks
