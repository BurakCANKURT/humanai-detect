"""QuillBot / Wordtune / ParrotAI, geri-ceviri (back-translation) veya bir LLM ile ham
yapay metinlerin 'humanize' edilmesi.

Uc yol var:
1. Manuel: QuillBot/Wordtune/ParrotAI'nin genel kullanima acik API'si olmadigi icin,
   ham-AI (ai_raw) metinler aracin web arayuzunde elle yeniden yazilir ve
   data/external/humanized/<tool>/<sample_id>.txt olarak kaydedilir; import_humanized_sample/
   humanize_batch bu klasoru okuyup orijinal ai_raw ornegiyle id'den eslestirir.
2. Otomatik (LLM): Gemini gibi bir LLM, ai_raw metnini "daha insansi" gorunecek sekilde
   yeniden yazar (humanize_with_gemini/humanize_batch_llm) - manuel dosya kopyalama gerekmez.
3. Otomatik (geri-ceviri): Turkce->Ingilizce->Turkce ceviri (Helsinki-NLP/OPUS-MT,
   humanize_with_backtranslation) ile paraphrase uretir. QuillBot/Wordtune/ParrotAI'nin
   API'sizlik/dil-destegi kisitlari nedeniyle pratik bir alternatif olarak eklendi;
   metodolojik olarak DIPPER'in (Krishna ve ark., NeurIPS 2023, "Paraphrasing evades
   detectors of AI-generated text") paraphrase-tabanli insansilastirma yaklasimindan
   esinlenmistir. Ayri bir mimari (seq2seq ceviri modeli) kullandigi icin ai_raw uretiminde
   kullanilan LLM ile ayni model olma sorunu yasanmaz.

Hangi araclarin/saglayicilarin aktif oldugu configs/data_sources.yaml -> humanizers.*
altinda tanimlidir.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from tenacity import Retrying, stop_after_attempt, wait_exponential

from .schemas import RawSample

_HUMANIZE_PROMPT_TEMPLATE = (
    "Asagidaki yapay zeka tarafindan uretilmis metni, anlamini ve icerigini degistirmeden "
    "daha dogal, akici ve bir insan tarafindan yazilmis gibi hissettirecek sekilde yeniden "
    "yaz. Cumle yapilarini cesitlendir, tekrar eden kaliplari kaldir. "
    "Yeniden yazdigin metin yaklasik {word_count} kelime olmali (gerekirse orijinal metinden "
    "daha uzun yaz, ONEMLI OLCUDE KISALTMA YAPMA). "
    "Sadece yeniden yazilmis metni dondur, aciklama ekleme.\n\n"
    "Metin:\n{text}"
)


def import_humanized_sample(ai_raw_sample: RawSample, tool: str, source_dir: Path) -> RawSample | None:
    """Bir ai_raw orneği icin elle humanize edilmis karsiligini <source_dir>/<id>.txt'den okur."""
    path = source_dir / f"{ai_raw_sample.id}.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return RawSample(
        id=f"ai_humanized_{tool}_{ai_raw_sample.id}",
        text=text,
        label="ai_humanized",
        source=tool,
        metadata={"original_id": ai_raw_sample.id, "original_source": ai_raw_sample.source},
    )


def humanize_batch(ai_raw_samples: list[RawSample], tool: str, source_dir: Path) -> list[RawSample]:
    """ai_raw orneklerinin <source_dir> altinda elle humanize edilmis karsiliklarini toplar.

    Henuz <source_dir>'a humanize edilmis dosyasi konmamis ornekler atlanir; bu fonksiyon
    kullanicinin o ana kadar elle tamamladigi kismi sonucu dondurur.
    """
    samples: list[RawSample] = []
    for ai_raw_sample in ai_raw_samples:
        humanized = import_humanized_sample(ai_raw_sample, tool, source_dir)
        if humanized is not None:
            samples.append(humanized)
    return samples


def humanize_with_gemini(text: str, model: str, api_key: str) -> str:
    """Google Gemini ile bir ai_raw metnini otomatik humanize eder."""
    from google import genai
    from .llm_generators import _sample_target_words

    client = genai.Client(api_key=api_key)
    word_count = _sample_target_words(random.Random())
    response = client.models.generate_content(
        model=model, contents=_HUMANIZE_PROMPT_TEMPLATE.format(text=text, word_count=word_count)
    )
    return response.text or ""


def humanize_with_openai(text: str, model: str, api_key: str, **length_kwargs) -> str:
    """GPT-4o-mini ile bir ai_raw metnini otomatik humanize eder (prompt-tabanli yeniden yazim,
    back-translation DEGIL).

    Eklenme sebebi (bkz. proje notlari, 2026-07-21): kisa-metin (5-30 kelime) havuzunda
    back-translation'in neredeyse hic degisiklik yapmadigi olculdu (raw<->humanized char-benzerligi
    ort=0.854, ana havuzdaki ort=0.17-0.26'nin cok uzerinde) -- round-trip cevirinin kisa/tek
    cumlelik metinde neredeyse tersine-cevrilebilir olmasi. Kisa-metin havuzu icin bu fonksiyon
    kullanilir (bkz. scripts/humanize_short_openai.py), ana havuz icin back-translation
    (humanize_with_backtranslation) degismeden kullanilmaya devam eder (orada ayrim yeterliydi).

    length_kwargs: _sample_target_words'e dogrudan gecilir (mean/std/min_words/max_words) --
    kisa-metin cagrisinda ZORUNLU olarak short_pilot'un 5-30 kelime araligiyla override edilmeli,
    aksi halde modulun varsayilan (uzun-form, ort=1750) hedefi kullanilir.
    """
    from openai import OpenAI
    from .llm_generators import _sample_target_words

    client = OpenAI(api_key=api_key, timeout=60.0)
    word_count = _sample_target_words(random.Random(), **length_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _HUMANIZE_PROMPT_TEMPLATE.format(text=text, word_count=word_count)}],
    )
    return response.choices[0].message.content or ""


def humanize_with_llama(text: str, model: str, endpoint: str, api_key: str | None = None) -> str:
    """Yerel/self-hosted bir LLM (orn. Ollama) ile bir ai_raw metnini otomatik humanize eder.

    API kota/kapasite sorunlarindan bagimsizdir (lokal calisir, ucretsizdir).
    """
    import httpx
    from .llm_generators import _sample_target_words

    word_count = _sample_target_words(random.Random())
    response = httpx.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        json={
            "model": model,
            "messages": [{"role": "user", "content": _HUMANIZE_PROMPT_TEMPLATE.format(text=text, word_count=word_count)}],
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def humanize_with_transformers(
    text: str,
    model_id: str,
    device: str = "auto",
    load_in_4bit: bool = True,
) -> str:
    """HuggingFace Transformers ile ai_raw metnini humanize eder (Colab/GPU ortami icin)."""
    from .llm_generators import _get_hf_pipeline, _sample_target_words

    pipe = _get_hf_pipeline(model_id, device, load_in_4bit)
    word_count = _sample_target_words(random.Random())
    messages = [{"role": "user", "content": _HUMANIZE_PROMPT_TEMPLATE.format(text=text, word_count=word_count)}]
    outputs = pipe(messages, max_new_tokens=1024, do_sample=True, temperature=0.7,
                   pad_token_id=pipe.tokenizer.eos_token_id)
    return outputs[0]["generated_text"][-1]["content"]


_LLM_HUMANIZERS = {
    "gemini": humanize_with_gemini,
    "llama": humanize_with_llama,
    "transformers": humanize_with_transformers,
    "openai": humanize_with_openai,
}

_LOCAL_PROVIDERS = frozenset({"transformers", "llama", "backtranslate"})

_MT_MODEL_CACHE: dict[str, tuple] = {}


def _get_mt_model(model_id: str, device: str):
    """Marian/seq2seq ceviri modelini (+tokenizer) yukler, cache'ler."""
    cache_key = f"{model_id}|{device}"
    if cache_key in _MT_MODEL_CACHE:
        return _MT_MODEL_CACHE[cache_key]

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(resolved_device)
    model.eval()
    _MT_MODEL_CACHE[cache_key] = (model, tokenizer, resolved_device)
    return _MT_MODEL_CACHE[cache_key]


def _translate_batch(texts: list[str], model_id: str, device: str, max_length: int = 512) -> list[str]:
    """Bir metin listesini tek bir yonde (orn. tr->en) topluca cevirir."""
    import torch

    model, tokenizer, resolved_device = _get_mt_model(model_id, device)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(resolved_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=4)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def _split_for_translation(text: str, max_words: int = 80) -> list[str]:
    """Uzun metni ceviri modelinin max_length sinirina (512 subword token) guvenle
    sigacak kucuk, cumle-sinirli parcalara boler."""
    from .file_ingest import chunk_text

    pieces = chunk_text(text, min_words=1, max_words=max_words)
    return pieces if pieces else [text]


def humanize_with_backtranslation(
    text: str,
    tr_en_model: str = "Helsinki-NLP/opus-mt-tc-big-tr-en",
    en_tr_model: str = "Helsinki-NLP/opus-mt-tc-big-en-tr",
    device: str = "auto",
) -> str:
    """Turkce->Ingilizce->Turkce geri-ceviri ile paraphrase uretir (bkz. modul docstring'i)."""
    tr_chunks = _split_for_translation(text)
    en_chunks = _translate_batch(tr_chunks, tr_en_model, device)
    tr_back_chunks = _translate_batch(en_chunks, en_tr_model, device)
    return " ".join(tr_back_chunks)


def _humanize_batch_backtranslation(
    ai_raw_samples: list[RawSample],
    start_index: int,
    checkpoint_path: Path | None,
    tr_en_model: str = "Helsinki-NLP/opus-mt-tc-big-tr-en",
    en_tr_model: str = "Helsinki-NLP/opus-mt-tc-big-en-tr",
    device: str = "auto",
    output_label: str = "ai_humanized",
    id_prefix: str = "ai_humanized_backtranslate",
) -> list[RawSample]:
    """output_label/id_prefix (bkz. proje notlari, 2026-07-21 -- DAMAGE makalesindeki
    'humanization invariance' bulgusu): varsayilan olarak ai_raw->ai_humanized icin
    kullanilir, ama AYNI fonksiyon insan metnini de back-translation'dan gecirip
    label DEGISTIRMEDEN ("human") augmentasyon olarak eklemek icin de kullanilabilir
    (bkz. scripts/humanize_human_topup.py) -- model boylece back-translation'in
    kendisinin bir AI imzasi OLMADIGINI, sadece ceviri artigini ogrenir."""
    from dataclasses import asdict

    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    samples_to_process = ai_raw_samples[start_index:]
    total = len(samples_to_process)
    samples: list[RawSample] = []

    for i, ai_raw_sample in enumerate(samples_to_process, start=1):
        print(f"  [{i}/{total}] backtranslate: {ai_raw_sample.id} isleniyor...", flush=True)
        text = humanize_with_backtranslation(ai_raw_sample.text, tr_en_model, en_tr_model, device).strip()
        if not text:
            print(f"  UYARI: bos yanit (ornek={ai_raw_sample.id}), atlaniyor.", flush=True)
            continue
        sample = RawSample(
            id=f"{id_prefix}_{ai_raw_sample.id}",
            text=text,
            label=output_label,
            source="backtranslate",
            metadata={
                "original_id": ai_raw_sample.id,
                "original_source": ai_raw_sample.source,
                "tr_en_model": tr_en_model,
                "en_tr_model": en_tr_model,
            },
        )
        samples.append(sample)
        if checkpoint_path is not None:
            with checkpoint_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")

    return samples


def _humanize_batch_transformers(
    ai_raw_samples: list[RawSample],
    provider: str,
    start_index: int,
    checkpoint_path: Path | None,
    batch_size: int = 8,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "auto",
    load_in_4bit: bool = True,
) -> list[RawSample]:
    """Transformers pipeline ile batch GPU inference yaparak humanize eder."""
    from dataclasses import asdict
    from .llm_generators import _MAX_NEW_TOKENS, _get_hf_pipeline, _sample_target_words

    pipe = _get_hf_pipeline(model_id, device, load_in_4bit)
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    samples_to_process = ai_raw_samples[start_index:]
    total = len(samples_to_process)
    n_batches = (total + batch_size - 1) // batch_size
    samples: list[RawSample] = []
    rng = random.Random(42)

    for b in range(n_batches):
        batch = samples_to_process[b * batch_size : (b + 1) * batch_size]
        messages_batch = [
            [{"role": "user", "content": _HUMANIZE_PROMPT_TEMPLATE.format(text=s.text, word_count=_sample_target_words(rng))}]
            for s in batch
        ]
        print(f"  [batch {b+1}/{n_batches}] {len(batch)} ornek humanize ediliyor...", flush=True)

        outputs = pipe(
            messages_batch,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=pipe.tokenizer.eos_token_id,
            batch_size=len(batch),
        )

        for k, ai_raw_sample in enumerate(batch):
            text = outputs[k][0]["generated_text"][-1]["content"].strip()
            if not text:
                print(f"  UYARI: bos yanit (ornek={ai_raw_sample.id}), atlaniyor.", flush=True)
                continue
            sample = RawSample(
                id=f"ai_humanized_{provider}_{ai_raw_sample.id}",
                text=text,
                label="ai_humanized",
                source=provider,
                metadata={
                    "original_id": ai_raw_sample.id,
                    "original_source": ai_raw_sample.source,
                    "model": model_id,
                },
            )
            samples.append(sample)
            if checkpoint_path is not None:
                with checkpoint_path.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")

        done = min((b + 1) * batch_size, total)
        print(f"  [{done}/{total}] tamamlandi.", flush=True)

    return samples


def humanize_batch_llm(
    ai_raw_samples: list[RawSample],
    provider: str,
    rate_limit: dict | None = None,
    checkpoint_path: Path | None = None,
    start_index: int = 0,
    max_concurrency: int = 1,
    **provider_kwargs,
) -> list[RawSample]:
    """ai_raw orneklerini bir LLM (orn. Gemini/Ollama) ile otomatik humanize eder, manuel dosya gerektirmez.

    checkpoint_path : her ornek uretilir uretilmez bu dosyaya JSON satiri eklenir (yeniden baslama destegi).
    start_index     : checkpoint'ten devam ederken atlanan ornek sayisi.
    provider_kwargs : secilen saglayicinin humanize_with_* fonksiyonuna gerekli parametreler.

    max_concurrency>1 (API saglayicilari icin): istekleri ThreadPoolExecutor ile paralel
    gonderir. Eklenme sebebi (bkz. proje notlari, 2026-07-25): uzun-metin (~650-1000 kelime)
    humanize cagrilarinda gercek darbogaz rate-limit degil, GPT-4o-mini'nin UZUN cikti
    URETME SURESI (~15-25sn/cagri) -- sirali calisirken gozlemlenen gercek hiz sadece
    ~2 ornek/dakika (2812 ornek icin ~23 saat), 8-10 RPM varsayimiyla tahmin edilen ~5
    saatin cok uzerinde. generate_batch'teki (llm_generators.py) ayni deseni kullanir
    (_RateLimiter, gercek RPM tavanini asmayan thread-safe kayan-pencere limiter).
    """
    # Transformers için doğrudan batch inference kullan
    if provider == "transformers":
        batch_size = int(provider_kwargs.pop("batch_size", 8))
        return _humanize_batch_transformers(
            ai_raw_samples, provider, start_index, checkpoint_path,
            batch_size=batch_size, **provider_kwargs,
        )

    if provider == "backtranslate":
        return _humanize_batch_backtranslation(
            ai_raw_samples, start_index, checkpoint_path, **provider_kwargs,
        )

    humanize_fn = _LLM_HUMANIZERS[provider]
    rate_limit = rate_limit or {}
    max_retries = rate_limit.get("max_retries", 3)
    requests_per_minute = rate_limit.get("requests_per_minute", 60)
    retryer = Retrying(stop=stop_after_attempt(max_retries), wait=wait_exponential(multiplier=1, max=30))

    total = len(ai_raw_samples)
    remaining = ai_raw_samples[start_index:]

    if max_concurrency > 1 and provider not in _LOCAL_PROVIDERS:
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from .llm_generators import _RateLimiter

        limiter = _RateLimiter(requests_per_minute)
        write_lock = threading.Lock()
        done_count = 0
        samples: list[RawSample] = []

        def _worker(idx: int, ai_raw_sample: RawSample) -> tuple[int, RawSample, str]:
            limiter.acquire()
            try:
                text = retryer(humanize_fn, ai_raw_sample.text, **provider_kwargs).strip()
            except Exception as exc:
                print(f"  UYARI: id={ai_raw_sample.id} icin kalici hata ({exc!r}), atlaniyor.", flush=True)
                text = ""
            return idx, ai_raw_sample, text

        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = {
                pool.submit(_worker, start_index + j, s): start_index + j
                for j, s in enumerate(remaining)
            }
            for future in as_completed(futures):
                idx, ai_raw_sample, text = future.result()
                with write_lock:
                    done_count += 1
                    if text:
                        sample = RawSample(
                            id=f"ai_humanized_{provider}_{ai_raw_sample.id}",
                            text=text,
                            label="ai_humanized",
                            source=provider,
                            metadata={
                                "original_id": ai_raw_sample.id,
                                "original_source": ai_raw_sample.source,
                                "model": provider_kwargs.get("model", ""),
                            },
                        )
                        samples.append(sample)
                        if checkpoint_path is not None:
                            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                            with checkpoint_path.open("a", encoding="utf-8") as fp:
                                from dataclasses import asdict
                                fp.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
                        print(f"[{done_count}/{len(remaining)}] tamam — id={sample.id}", flush=True)
                    else:
                        print(f"[{done_count}/{len(remaining)}] UYARI: bos yanit (id={ai_raw_sample.id}), atlaniyor.", flush=True)
        return samples

    # Yerel sağlayıcılar (llama/ollama) için rate limit bekleme gereksiz
    min_interval = 0.0 if provider in _LOCAL_PROVIDERS else (60.0 / requests_per_minute if requests_per_minute else 0.0)
    samples = []
    for i, ai_raw_sample in enumerate(ai_raw_samples[start_index:], start=start_index):
        global_num = i + 1
        print(f"[{global_num}/{total}] {provider}: humanize isleniyor (ornek #{global_num})...")
        text = retryer(humanize_fn, ai_raw_sample.text, **provider_kwargs).strip()
        if text:
            sample = RawSample(
                id=f"ai_humanized_{provider}_{ai_raw_sample.id}",
                text=text,
                label="ai_humanized",
                source=provider,
                metadata={
                    "original_id": ai_raw_sample.id,
                    "original_source": ai_raw_sample.source,
                    "model": provider_kwargs.get("model", ""),
                },
            )
            samples.append(sample)
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint_path.open("a", encoding="utf-8") as fp:
                    from dataclasses import asdict
                    fp.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
            print(f"[{global_num}/{total}] tamam — {len(text)} karakter yazildi")
        if min_interval and i < total - 1:
            time.sleep(min_interval)
    return samples
