"""Stanza tabanli cumle bolme, tokenizasyon, POS etiketleme ve dependency parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import stanza as _stanza_type

_NLP: "_stanza_type.Pipeline | None" = None


def _get_nlp() -> "_stanza_type.Pipeline":
    """Stanza Turkce pipeline'ini tembel yukler (ilk cagirda indirir)."""
    global _NLP
    if _NLP is None:
        import stanza

        # Model zaten indirildiyse GitHub kontrolu atlaniyor (ag kesintisine karsi)
        try:
            stanza.download("tr", processors="tokenize,pos,lemma,depparse", verbose=False)
        except Exception:
            pass  # Model yerel olarak mevcut, ag hatasi gorumsezden geliniyor

        _NLP = stanza.Pipeline(
            "tr",
            processors="tokenize,pos,lemma,depparse",
            use_gpu=False,
            verbose=False,
        )
    return _NLP


def analyze(text: str) -> dict:
    """Tek Stanza gecisinden cumle, token, POS ve dep parse bilgilerini dondurur.

    Donus sozlugu:
        sentences  : list[str]              -- cumle metinleri
        tokens     : list[str]              -- duz token listesi
        pos_tags   : list[tuple[str,str]]   -- (token, UPOS) ciftleri
        dep_parse  : list[dict]             -- id/text/lemma/upos/head/deprel sozlukleri
    """
    nlp = _get_nlp()
    doc = nlp(text)

    sentences = [s.text for s in doc.sentences]
    tokens: list[str] = []
    pos_tags: list[tuple[str, str]] = []
    dep_parse: list[dict] = []

    for sent in doc.sentences:
        for word in sent.words:
            tokens.append(word.text)
            pos_tags.append((word.text, word.upos or ""))
            dep_parse.append(
                {
                    "id": word.id,
                    "text": word.text,
                    "lemma": word.lemma or word.text,
                    "upos": word.upos or "",
                    "head": word.head,
                    "deprel": word.deprel or "",
                }
            )

    return {
        "sentences": sentences,
        "tokens": tokens,
        "pos_tags": pos_tags,
        "dep_parse": dep_parse,
    }


def pos_tag(text: str) -> list[tuple[str, str]]:
    """Metin icin (token, UPOS) ciftleri dondurur."""
    return analyze(text)["pos_tags"]


def dependency_parse(text: str) -> list[dict]:
    """Metin icin per-word depparse kayitlari dondurur."""
    return analyze(text)["dep_parse"]
