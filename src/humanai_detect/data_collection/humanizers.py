"""QuillBot / Wordtune / ParrotAI veya bir LLM ile ham yapay metinlerin 'humanize' edilmesi.

Iki yol var:
1. Manuel: QuillBot/Wordtune/ParrotAI'nin genel kullanima acik API'si olmadigi icin,
   ham-AI (ai_raw) metinler aracin web arayuzunde elle yeniden yazilir ve
   data/external/humanized/<tool>/<sample_id>.txt olarak kaydedilir; import_humanized_sample/
   humanize_batch bu klasoru okuyup orijinal ai_raw ornegiyle id'den eslestirir.
2. Otomatik (LLM): Gemini gibi bir LLM, ai_raw metnini "daha insansi" gorunecek sekilde
   yeniden yazar (humanize_with_gemini/humanize_batch_llm) - manuel dosya kopyalama gerekmez.

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
}

_LOCAL_PROVIDERS = frozenset({"transformers", "llama"})


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
    **provider_kwargs,
) -> list[RawSample]:
    """ai_raw orneklerini bir LLM (orn. Gemini/Ollama) ile otomatik humanize eder, manuel dosya gerektirmez.

    checkpoint_path : her ornek uretilir uretilmez bu dosyaya JSON satiri eklenir (yeniden baslama destegi).
    start_index     : checkpoint'ten devam ederken atlanan ornek sayisi.
    provider_kwargs : secilen saglayicinin humanize_with_* fonksiyonuna gerekli parametreler.
    """
    # Transformers için doğrudan batch inference kullan
    if provider == "transformers":
        batch_size = int(provider_kwargs.pop("batch_size", 8))
        return _humanize_batch_transformers(
            ai_raw_samples, provider, start_index, checkpoint_path,
            batch_size=batch_size, **provider_kwargs,
        )

    humanize_fn = _LLM_HUMANIZERS[provider]
    rate_limit = rate_limit or {}
    max_retries = rate_limit.get("max_retries", 3)
    requests_per_minute = rate_limit.get("requests_per_minute", 60)
    # Yerel sağlayıcılar (llama/ollama) için rate limit bekleme gereksiz
    min_interval = 0.0 if provider in _LOCAL_PROVIDERS else (60.0 / requests_per_minute if requests_per_minute else 0.0)
    retryer = Retrying(stop=stop_after_attempt(max_retries), wait=wait_exponential(multiplier=1, max=30))

    total = len(ai_raw_samples)
    samples: list[RawSample] = []
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
