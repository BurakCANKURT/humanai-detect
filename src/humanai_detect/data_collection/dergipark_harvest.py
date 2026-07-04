"""DergiPark OAI-PMH ucundan Turkce (en gec 2019) acik erisim makale toplama.

DergiPark (TUBITAK ULAKBIM) resmi bir OAI-PMH metadata-harvesting ucu sunar
(https://dergipark.org.tr/api/public/oai/); YOK Tez Merkezi'nin aksine CAPTCHA/bot
korumasi yok, robots.txt de /search ve /login disinda serbest (bkz. human_sources.py'deki
YOK Tez Merkezi notuyla karsilastir). Tam metin PDF'i makale sayfasindaki dogrudan
indirme linkinden alinir.

Bazi dergilerin PDF'lerinde gomulu fontun ToUnicode CMap'i eksik/bozuk oldugu icin
metin cikarimi anlamsiz karakter dizisine donusebiliyor (font-duzeyinde, tek makaleye
ozgu bir sorun). Bu yuzden her cikarilan metin bir Turkce durak-kelime orani ile
dogrulanir; esigin altinda kalanlar atilir.

dc:language metadata alani 'tr' dese de gercek icerik Ingilizce olabilir (yanlis
etiketlenmis kayitlar, tam Ingilizce makaleler vb.); bu yuzden metin ayrica py3langid
ile dil tespitinden gecirilir.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Iterable

import fitz  # pymupdf
import py3langid as langid

from .file_ingest import chunk_text
from .schemas import RawSample

_OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/"}
_DC_NS = {"dc": "http://purl.org/dc/elements/1.1/"}
_OAI_DC_TAG = "{http://www.openarchives.org/OAI/2.0/oai_dc/}dc"
_OAI_BASE = "https://dergipark.org.tr/api/public/oai/"
_UA = "Mozilla/5.0 (compatible; ADU-TUBITAK-2209A-research-collector/0.1)"
_MAX_YEAR = 2019  # en gec 2019 (2019 ve oncesi) yayinlar hedefleniyor
_REQUEST_DELAY = 12.0
_VALIDITY_THRESHOLD = 0.03
_PDF_LINK_RE = re.compile(r"download/article-file/\d+")
_JOURNAL_URL_RE = re.compile(r"/pub/([^/]+)/article/")
_DEFAULT_MAX_PER_JOURNAL = 5  # tek dergiden asiri agirlik almayi onlemek icin (stilometri cesitliligi)
_JOURNAL_GIVE_UP_STREAK = 6  # bir dergiden ust uste bu kadar red gelirse (tarama/eski PDF sorunu) o dergi atlanir

_STOPWORDS = {
    "ve", "bir", "bu", "için", "ile", "de", "da", "olarak", "olan", "gibi",
    "en", "çok", "daha", "ancak", "ise", "ama", "veya", "ki", "mi", "mu",
    "ne", "her", "tüm", "sonra", "önce", "kadar", "göre", "ya", "yani",
}


def _fetch(url: str, max_retries: int = 6) -> bytes:
    """429/gecici hatalarda ustel geri cekilme ile getirir.

    Windows'ta Python'in urllib'i bazi DergiPark isteklerinde (nedeni belirsiz,
    muhtemelen TLS/soket duzeyinde) suresiz askida kalabiliyordu -- ayni URL'ler
    curl ile her zaman aninda yanit veriyordu. Bu yuzden fetch, subprocess.run'in
    guvenilir sekilde uyguladigi sabit bir 'timeout' ile curl'e devredildi.
    """
    delay = 5.0
    for attempt in range(max_retries):
        fd, tmp_path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        try:
            proc = subprocess.run(
                ["curl", "-sL", "--max-time", "20", "-A", _UA, "-o", tmp_path, "-w", "%{http_code}", url],
                capture_output=True, timeout=30, text=True,
            )
            status = proc.stdout.strip()
            if status.startswith("2"):
                with open(tmp_path, "rb") as f:
                    return f.read()
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"HTTP {status}: {url}")
        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"curl zaman asimi: {url}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    raise RuntimeError(f"fetch basarisiz: {url}")


def _oai_page(resumption_token: str | None) -> tuple[list[ET.Element], str | None]:
    if resumption_token:
        url = f"{_OAI_BASE}?verb=ListRecords&resumptionToken={resumption_token}"
    else:
        # 'from' OAI repository datestamp'ine (son degisiklik) gore filtreler, yayin
        # yilina gore degil (bkz. dc:date client-side filtresi asagida) -- eski
        # (<=2019) yayinlari da kapsamak icin ust sinir konmuyor.
        url = f"{_OAI_BASE}?verb=ListRecords&metadataPrefix=oai_dc"
    root = ET.fromstring(_fetch(url))
    records = root.findall(".//oai:record", _OAI_NS)
    token_elem = root.find(".//oai:resumptionToken", _OAI_NS)
    next_token = token_elem.text if token_elem is not None and token_elem.text else None
    return records, next_token


def _parse_candidate(rec: ET.Element) -> dict | None:
    header = rec.find("oai:header", _OAI_NS)
    if header is not None and header.get("status") == "deleted":
        return None
    dc = rec.find(f".//{_OAI_DC_TAG}")
    if dc is None:
        return None

    lang = (dc.findtext("dc:language", default="", namespaces=_DC_NS) or "").strip()
    date = (dc.findtext("dc:date", default="", namespaces=_DC_NS) or "").strip()
    dtype = dc.findtext("dc:type", default="", namespaces=_DC_NS) or ""
    title = (dc.findtext("dc:title", default="", namespaces=_DC_NS) or "").strip()
    identifiers = [e.text for e in dc.findall("dc:identifier", _DC_NS) if e.text]
    article_url = next(
        (i for i in identifiers if "dergipark.org.tr" in i and "/article/" in i), None
    )

    if lang != "tr" or "article" not in dtype or not article_url:
        return None
    if not date[:4].isdigit() or int(date[:4]) > _MAX_YEAR:
        return None

    oai_id = header.findtext("oai:identifier", default=article_url, namespaces=_OAI_NS)
    journal_m = _JOURNAL_URL_RE.search(article_url)
    journal = journal_m.group(1) if journal_m else "unknown"
    return {"oai_id": oai_id, "article_url": article_url, "title": title, "date": date, "journal": journal}


def _get_pdf_link(article_url: str) -> str | None:
    html = _fetch(article_url).decode("utf-8", errors="ignore")
    m = _PDF_LINK_RE.search(html)
    if not m:
        return None
    base = article_url.split("/en/")[0].split("/tr/")[0]
    return f"{base}/en/{m.group(0)}"


def _turkish_validity_ratio(text: str) -> float:
    words = re.findall(r"[a-zA-ZçÇğĞıİöÖşŞüÜ]+", text.lower())
    if len(words) < 30:
        return 0.0
    hits = sum(1 for w in words if w in _STOPWORDS)
    return hits / len(words)


def _is_turkish(text: str) -> bool:
    lang, _ = langid.classify(text)
    return lang == "tr"


_BIBLIOGRAPHY_LINE_RE = re.compile(r"\(\d{4}\)|\d{4}\)\.|pp\.\s*\d|ss\.\s*\d")
_BIBLIOGRAPHY_RATIO_THRESHOLD = 0.35


def _bibliography_ratio(text: str) -> float:
    """Chunk'in ne kadarinin kaynakca/referans listesi gibi gorundugunu olcer.

    Stilometri icin onemli: kaynakca bolumleri duz anlatim degildir ve AI-uretilen
    sinifta hemen hic bulunmaz; bu yuzden fazla kaynakca-agirlikli chunk'lar
    modelin gercek uslup farkindan degil bu yapay ipucundan ogrenmesine yol acabilir.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not lines:
        return 0.0
    hits = sum(1 for l in lines if _BIBLIOGRAPHY_LINE_RE.search(l))
    return hits / len(lines)


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.setdefault("journal_counts", {})
        state.setdefault("journal_reject_streak", {})
        state.setdefault("journal_blacklist", [])
        return state
    return {
        "resumption_token": None, "exhausted": False, "rejected_ids": [],
        "journal_counts": {}, "journal_reject_streak": {}, "journal_blacklist": [],
    }


def _save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def harvest(
    accepted_article_ids: set[str],
    target_new_chunks: int,
    min_tokens: int,
    max_tokens: int,
    state_path: Path,
    on_chunk: Callable[[RawSample], None],
    journal_counts: dict[str, int] | None = None,
    max_per_journal: int = _DEFAULT_MAX_PER_JOURNAL,
) -> dict:
    """Hedefe ulasana ya da katalog tukenene kadar DergiPark'i tarar.

    Her kabul edilen chunk `on_chunk` ile aninda disariya (checkpoint) yazdirilir.
    `state_path` OAI sayfalama konumunu ve reddedilen makaleleri kalici tutar, boylece
    kesintiye ugrayan bir calisma tekrar baslatildiginda katalogu bastan taramaz.
    `journal_counts` (cagiran taraftan, mevcut kabul edilmis kayitlardan hesaplanir)
    ve `max_per_journal` tek bir dergiden asiri agirlik alinmasini onler (stilometri
    cesitliligi icin).
    """
    state = _load_state(state_path)
    rejected_ids: set[str] = set(state.get("rejected_ids", []))
    journal_counts = dict(journal_counts) if journal_counts else dict(state.get("journal_counts", {}))
    journal_reject_streak: dict[str, int] = dict(state.get("journal_reject_streak", {}))
    journal_blacklist: set[str] = set(state.get("journal_blacklist", []))
    stats = {
        "attempted": 0, "accepted_chunks": 0, "garbled": 0, "not_turkish": 0,
        "no_pdf_link": 0, "too_short": 0, "fetch_error": 0, "journal_capped": 0,
        "journal_blacklisted": 0, "bibliography_chunk": 0,
    }

    def _reject(oai_id: str, journal: str) -> None:
        rejected_ids.add(oai_id)
        streak = journal_reject_streak.get(journal, 0) + 1
        journal_reject_streak[journal] = streak
        if streak >= _JOURNAL_GIVE_UP_STREAK:
            journal_blacklist.add(journal)

    if state.get("exhausted"):
        return stats

    while stats["accepted_chunks"] < target_new_chunks:
        for page_attempt in range(3):
            try:
                records, next_token = _oai_page(state.get("resumption_token"))
                break
            except Exception:
                if page_attempt == 2:
                    raise
                time.sleep(10.0)
        for rec in records:
            if stats["accepted_chunks"] >= target_new_chunks:
                break
            cand = _parse_candidate(rec)
            if cand is None:
                continue
            oai_id = cand["oai_id"]
            if oai_id in accepted_article_ids or oai_id in rejected_ids:
                continue
            if journal_counts.get(cand["journal"], 0) >= max_per_journal:
                stats["journal_capped"] += 1
                continue
            if cand["journal"] in journal_blacklist:
                stats["journal_blacklisted"] += 1
                continue

            stats["attempted"] += 1
            try:
                pdf_link = _get_pdf_link(cand["article_url"])
                time.sleep(_REQUEST_DELAY)
                if not pdf_link:
                    stats["no_pdf_link"] += 1
                    _reject(oai_id, cand["journal"])
                    continue

                pdf_bytes = _fetch(pdf_link)
                time.sleep(_REQUEST_DELAY)
                text = _extract_pdf_text(pdf_bytes)

                if _turkish_validity_ratio(text) < _VALIDITY_THRESHOLD:
                    stats["garbled"] += 1
                    _reject(oai_id, cand["journal"])
                    continue

                if not _is_turkish(text):
                    stats["not_turkish"] += 1
                    _reject(oai_id, cand["journal"])
                    continue

                chunks = chunk_text(text, min_tokens, max_tokens)
                before = len(chunks)
                chunks = [c for c in chunks if _bibliography_ratio(c) <= _BIBLIOGRAPHY_RATIO_THRESHOLD]
                stats["bibliography_chunk"] += before - len(chunks)
                if not chunks:
                    stats["too_short"] += 1
                    _reject(oai_id, cand["journal"])
                    continue

                for chunk in chunks:
                    if stats["accepted_chunks"] >= target_new_chunks:
                        break
                    on_chunk(
                        RawSample(
                            id="",
                            text=chunk,
                            label="human",
                            source="dergipark",
                            metadata={
                                "article_url": cand["article_url"],
                                "title": cand["title"],
                                "date": cand["date"],
                                "oai_id": oai_id,
                                "journal": cand["journal"],
                            },
                        )
                    )
                    stats["accepted_chunks"] += 1
                accepted_article_ids.add(oai_id)
                journal_counts[cand["journal"]] = journal_counts.get(cand["journal"], 0) + 1
                journal_reject_streak[cand["journal"]] = 0
            except Exception:
                stats["fetch_error"] += 1
                time.sleep(_REQUEST_DELAY)

            state["rejected_ids"] = sorted(rejected_ids)
            state["journal_counts"] = journal_counts
            state["journal_reject_streak"] = journal_reject_streak
            state["journal_blacklist"] = sorted(journal_blacklist)
            _save_state(state_path, state)

        state["resumption_token"] = next_token
        state["exhausted"] = next_token is None
        state["rejected_ids"] = sorted(rejected_ids)
        state["journal_counts"] = journal_counts
        state["journal_reject_streak"] = journal_reject_streak
        state["journal_blacklist"] = sorted(journal_blacklist)
        _save_state(state_path, state)
        if next_token is None:
            break

    return stats
