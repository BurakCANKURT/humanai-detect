from __future__ import annotations

from pathlib import Path

from humanai_detect.data_collection import humanizers as humanizers_module
from humanai_detect.data_collection.corpus_download import download_turkish_wikipedia
from humanai_detect.data_collection.file_ingest import chunk_text
from humanai_detect.data_collection.humanizers import humanize_batch, humanize_batch_llm
from humanai_detect.data_collection.human_sources import collect_from_manual_corpus
from humanai_detect.data_collection.llm_generators import load_prompts
from humanai_detect.data_collection.schemas import RawSample
from humanai_detect.utils.io import read_jsonl, write_jsonl


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    records = [{"id": "a", "text": "merhaba", "label": "human"}, {"id": "b", "text": "selam", "label": "human"}]

    write_jsonl(path, records)
    result = read_jsonl(path)

    assert result == records


def test_collect_from_manual_corpus_reads_txt_files(tmp_path: Path) -> None:
    (tmp_path / "ornek1.txt").write_text("Bu bir insan yazimi metindir.", encoding="utf-8")
    (tmp_path / "ornek2.txt").write_text("Bu da ikinci ornek metindir.", encoding="utf-8")

    samples = collect_from_manual_corpus(target_count=10, source_dir=tmp_path)

    assert len(samples) == 2
    assert all(s.label == "human" and s.source == "manual_corpus" for s in samples)
    assert {s.text for s in samples} == {"Bu bir insan yazimi metindir.", "Bu da ikinci ornek metindir."}


def test_collect_from_manual_corpus_respects_target_count(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"ornek{i}.txt").write_text(f"metin {i}", encoding="utf-8")

    samples = collect_from_manual_corpus(target_count=2, source_dir=tmp_path)

    assert len(samples) == 2


def test_collect_from_manual_corpus_chunks_long_file(tmp_path: Path) -> None:
    # configs/preprocessing.yaml -> max_tokens: 2000; 1000 cumle * 3 kelime = 3000 kelime,
    # tek parcaya sigmaz ve en az 2 ornege bolunmesi gerekir.
    long_text = " ".join(f"Bu cumle {i}." for i in range(1000))
    (tmp_path / "uzun_tez.txt").write_text(long_text, encoding="utf-8")

    samples = collect_from_manual_corpus(target_count=10, source_dir=tmp_path)

    assert len(samples) > 1
    assert all(s.metadata["filename"] == "uzun_tez.txt" for s in samples)
    assert {s.id for s in samples} == {f"human_manual_corpus_{i:04d}" for i in range(len(samples))}


def test_chunk_text_splits_on_sentence_boundaries_within_max_words() -> None:
    text = " ".join(f"Cumle numara {i}." for i in range(100))

    chunks = chunk_text(text, min_words=10, max_words=50)

    assert len(chunks) > 1
    assert all(len(c.split()) <= 50 + 3 for c in chunks)  # son cumle tasmasi icin kucuk tolerans


def test_chunk_text_merges_short_trailing_chunk() -> None:
    text = "Cumle bir iki uc dort bes alti yedi sekiz dokuz on. Kisa son cumle."

    chunks = chunk_text(text, min_words=5, max_words=10)

    assert len(chunks) == 1


def test_chunk_text_returns_empty_list_for_blank_text() -> None:
    assert chunk_text("   ", min_words=10, max_words=100) == []


def test_humanize_batch_matches_by_id(tmp_path: Path) -> None:
    ai_raw_samples = [
        RawSample(id="ai_raw_openai_0000", text="orijinal metin", label="ai_raw", source="openai"),
        RawSample(id="ai_raw_openai_0001", text="ikinci metin", label="ai_raw", source="openai"),
    ]
    (tmp_path / "ai_raw_openai_0000.txt").write_text("yeniden yazilmis metin", encoding="utf-8")
    # ai_raw_openai_0001 icin dosya yok -> elenmeli

    samples = humanize_batch(ai_raw_samples, tool="quillbot", source_dir=tmp_path)

    assert len(samples) == 1
    assert samples[0].text == "yeniden yazilmis metin"
    assert samples[0].metadata["original_id"] == "ai_raw_openai_0000"


def test_humanize_batch_llm_calls_provider_for_each_sample(monkeypatch) -> None:
    calls = []

    def fake_humanize(text: str, model: str, api_key: str) -> str:
        calls.append((text, model, api_key))
        return f"humanized: {text}"

    monkeypatch.setitem(humanizers_module._LLM_HUMANIZERS, "gemini", fake_humanize)
    ai_raw_samples = [
        RawSample(id="ai_raw_gemini_0000", text="orijinal metin", label="ai_raw", source="gemini"),
    ]

    samples = humanize_batch_llm(ai_raw_samples, provider="gemini", model="gemini-1.5-pro", api_key="dummy")

    assert len(calls) == 1
    assert len(samples) == 1
    assert samples[0].text == "humanized: orijinal metin"
    assert samples[0].id == "ai_humanized_gemini_ai_raw_gemini_0000"
    assert samples[0].metadata["original_id"] == "ai_raw_gemini_0000"


def test_humanize_batch_backtranslation_roundtrips_via_translate_batch(monkeypatch) -> None:
    calls = []

    def fake_translate_batch(texts, model_id, device, max_length=512):
        calls.append((list(texts), model_id))
        return [f"[{model_id}] {t}" for t in texts]

    monkeypatch.setattr(humanizers_module, "_translate_batch", fake_translate_batch)
    ai_raw_samples = [
        RawSample(id="ai_raw_transformers_0000", text="kisa bir metin.", label="ai_raw", source="transformers"),
    ]

    samples = humanize_batch_llm(
        ai_raw_samples,
        provider="backtranslate",
        tr_en_model="tr-en-model",
        en_tr_model="en-tr-model",
    )

    assert len(calls) == 2
    assert calls[0][1] == "tr-en-model"
    assert calls[1][1] == "en-tr-model"
    assert len(samples) == 1
    assert samples[0].id == "ai_humanized_backtranslate_ai_raw_transformers_0000"
    assert samples[0].label == "ai_humanized"
    assert samples[0].source == "backtranslate"
    assert samples[0].metadata["original_id"] == "ai_raw_transformers_0000"
    assert "[en-tr-model]" in samples[0].text


def test_download_turkish_wikipedia_writes_files_and_skips_short_articles(tmp_path: Path) -> None:
    fake_articles = [
        {"title": "Kisa Madde", "text": "cok kisa"},
        {"title": "Uzun Madde Bir", "text": " ".join(["kelime"] * 50)},
        {"title": "Uzun Madde Iki", "text": " ".join(["kelime"] * 50)},
    ]

    written = download_turkish_wikipedia(
        output_dir=tmp_path, target_count=10, min_words=30, dataset=fake_articles
    )

    assert written == 2
    txt_files = sorted(tmp_path.glob("*.txt"))
    assert len(txt_files) == 2
    assert "Uzun_Madde_Bir" in txt_files[0].name


def test_download_turkish_wikipedia_respects_target_count(tmp_path: Path) -> None:
    fake_articles = [{"title": f"Madde {i}", "text": " ".join(["kelime"] * 50)} for i in range(10)]

    written = download_turkish_wikipedia(
        output_dir=tmp_path, target_count=3, min_words=30, dataset=fake_articles
    )

    assert written == 3
    assert len(list(tmp_path.glob("*.txt"))) == 3


def test_load_prompts_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("# yorum\n\nilk prompt\nikinci prompt\n", encoding="utf-8")

    prompts = load_prompts(prompts_file)

    assert prompts == ["ilk prompt", "ikinci prompt"]


# ---------------------------------------------------------------------------
# _sample_target_words: kisa-pilot verisi icin parametrelestirme (bkz.
# scripts/collect_short_pilot.py, [[project-codebase]] 2026-07-16)
# ---------------------------------------------------------------------------

import random

from humanai_detect.data_collection.llm_generators import (
    _TARGET_LEN_MAX, _TARGET_LEN_MEAN, _TARGET_LEN_MIN, _TARGET_LEN_STD, _sample_target_words,
)


def test_sample_target_words_default_matches_main_dataset_range() -> None:
    rng = random.Random(0)
    for _ in range(50):
        val = _sample_target_words(rng)
        assert _TARGET_LEN_MIN <= val <= _TARGET_LEN_MAX


def test_sample_target_words_custom_range_for_short_pilot() -> None:
    rng = random.Random(0)
    for _ in range(50):
        val = _sample_target_words(rng, mean=15, std=6, min_words=5, max_words=30)
        assert 5 <= val <= 30


def test_sample_target_words_clips_to_custom_bounds() -> None:
    rng = random.Random(0)
    # std=0.001 ile ortalamaya cok yakin degerler uretilir, min/max'in ISE
    # yaramasi icin ortalamayi sinirin disina koyuyoruz.
    val = _sample_target_words(rng, mean=1000, std=0.001, min_words=5, max_words=30)
    assert val == 30
