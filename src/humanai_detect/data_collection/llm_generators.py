"""GPT-4 / Gemini / Claude / Llama / HuggingFace Transformers ile ham yapay metin uretimi.

Hangi saglayicinin aktif oldugu configs/data_sources.yaml -> llm_generators.*.enabled
alaninda belirtilir; API key'ler .env'den (config.get_api_key) okunur.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from tenacity import Retrying, stop_after_attempt, wait_exponential

from .schemas import RawSample

_HF_PIPELINE_CACHE: dict[str, Any] = {}

_LOCAL_PROVIDERS = frozenset({"transformers", "llama"})


class _RateLimiter:
    """Kayan-pencere (sliding-window) rate limiter -- thread-safe, gercek RPM tavanini asmaz.

    Onceki yaklasim (sabit sure uyu, istek SONRASI) hem yavas (gecikme+uyku ust uste
    binerdi) hem de es zamanli (concurrent) cagrilarda hicbir korumasi yoktu -- 2026-07-19'da
    OpenAI hesabinin gercek limiti (10 RPM, "Limit 10, Used 10" 429 hatasi) concurrency=8 ile
    aninda asildi. Bu siniftaki acquire(), her cagriyi son 60 saniyedeki istek sayisi RPM
    tavanini gecmeyecek sekilde bloklar -- tek thread'de de, ThreadPoolExecutor icinde de guvenli.
    """

    def __init__(self, requests_per_minute: float | None) -> None:
        self.rpm = requests_per_minute
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def acquire(self) -> None:
        if not self.rpm:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60.0:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.rpm:
                    self._timestamps.append(now)
                    return
                wait_for = 60.0 - (now - self._timestamps[0]) + 0.05
            time.sleep(wait_for)

# Insan korpusunun GERCEK (3000 kayit, DergiPark harvester tamamlandiktan sonraki)
# kelime sayisi dagilimi: ort=1750, std=487, aralik=30-2026
# (bkz. data/raw/human/human.jsonl olcumu). Onceki 605/150/1200 degerleri, insan
# verisi henuz 500 kayitken yapilan eski bir olcumdeki (ort. 617) kalinti idi --
# 3000'e tamamlandiktan sonra guncellenmedigi icin AI/insan uzunluk farki ~3.3x'e
# cikmis ve confound payi (kelime-sayisi-tek-ozellik baseline / tam model) %98'e
# firlamisti (bkz. proje notlari, 2026-07-07). Simdi gercek dagilimla eslendi.
_TARGET_LEN_MEAN = 1750
_TARGET_LEN_STD = 487
_TARGET_LEN_MIN = 100
_TARGET_LEN_MAX = 2000
# Ortalama ~2.2 token/kelime (Turkce, Qwen2.5 tokenizer, gozlemlenen oran) --
# tavan, en uzun hedefi (2000 kelime) kirpmadan karsilayacak sekilde genis tutulur.
_MAX_NEW_TOKENS = 4500


def _sample_target_words(
    rng: random.Random,
    mean: float = _TARGET_LEN_MEAN,
    std: float = _TARGET_LEN_STD,
    min_words: int = _TARGET_LEN_MIN,
    max_words: int = _TARGET_LEN_MAX,
) -> int:
    val = rng.gauss(mean, std)
    return int(max(min_words, min(max_words, val)))


def _with_length_instruction(prompt: str, target_words: int) -> str:
    return f"{prompt}\n\nMetnin uzunlugu yaklasik {target_words} kelime olsun."


def load_prompts(path: Path) -> list[str]:
    """configs/prompts.txt'den yorum/bos satirlari atlayarak prompt listesi okur."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def generate_with_openai(prompt: str, model: str, api_key: str) -> str:
    """OpenAI (GPT-4) ile tek bir metin uretir."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=60.0)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def generate_with_gemini(prompt: str, model: str, api_key: str) -> str:
    """Google Gemini ile tek bir metin uretir."""
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or ""


def generate_with_anthropic(prompt: str, model: str, api_key: str) -> str:
    """Anthropic Claude ile tek bir metin uretir.

    thinking bilerek disabled -- bu duz bir metin-uretim gorevi (akademik deneme yazimi),
    akil yurutmeye ihtiyaci yok. Claude Sonnet 5'te thinking parametresi hic verilmezse
    adaptive thinking OTOMATIK devreye giriyor (sessiz varsayilan degisikligi) ve gorunmeyen
    ekstra output-token maliyeti dogurabiliyor -- bu, maliyeti hem daha yuksek hem daha
    ongorulemez yapiyordu (15 orneklik pilotta beklenenden ~%50 daha pahaliya cikti).

    max_tokens=8192 (eskiden 2048) -- ilk 49 ornekte (2026-07-19) max_tokens=2048'in
    Turkce metinde ~%84 oraninda yariden CUMLE ORTASINDA kesilmeye yol actigi bulundu
    (Turkce'nin agglutinatif yapisi/Claude tokenizer'i ~4+ token/kelime uretiyor, 2048
    token ~450-480 kelimede tukeniyor, hedef 680/1058 ne olursa olsun). O 49 ornek
    kullanilmadi, data/raw/_old_truncated_backup/'a tasindi.
    """
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key, timeout=90.0)
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate_with_llama(prompt: str, model: str, endpoint: str, api_key: str | None = None) -> str:
    """Yerel/self-hosted bir LLM (orn. Ollama) ile OpenAI-uyumlu chat-completions endpoint'i uzerinden metin uretir.

    Ollama icin endpoint tipik olarak http://localhost:11434/v1/chat/completions, api_key
    gerekmez (lokal calisir, ucretsizdir, API kota/kapasite sorunlarindan bagimsizdir).
    """
    import httpx

    response = httpx.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _get_hf_pipeline(model_id: str, device: str, load_in_4bit: bool) -> Any:
    """HuggingFace pipeline'ini yukler; ayni model tekrar tekrar yuklenmez."""
    cache_key = f"{model_id}|{device}|{load_in_4bit}"
    if cache_key in _HF_PIPELINE_CACHE:
        return _HF_PIPELINE_CACHE[cache_key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    print(f"[transformers] Model yukleniyor: {model_id} (load_in_4bit={load_in_4bit})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map=device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map=device)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    _HF_PIPELINE_CACHE[cache_key] = pipe
    print(f"[transformers] Model hazir: {model_id}")
    return pipe


def generate_with_transformers(
    prompt: str,
    model_id: str,
    device: str = "auto",
    load_in_4bit: bool = True,
) -> str:
    """HuggingFace Transformers ile metin uretir (Colab/GPU ortami icin, kota yok)."""
    pipe = _get_hf_pipeline(model_id, device, load_in_4bit)
    messages = [{"role": "user", "content": prompt}]
    outputs = pipe(messages, max_new_tokens=1024, do_sample=True, temperature=0.7,
                   pad_token_id=pipe.tokenizer.eos_token_id)
    return outputs[0]["generated_text"][-1]["content"]


_GENERATORS = {
    "openai": generate_with_openai,
    "gemini": generate_with_gemini,
    "anthropic": generate_with_anthropic,
    "llama": generate_with_llama,
    "transformers": generate_with_transformers,
}


def _generate_batch_transformers(
    prompts: list[str],
    provider: str,
    target_count: int,
    start_index: int,
    checkpoint_path: Path | None,
    batch_size: int = 8,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "auto",
    load_in_4bit: bool = True,
    target_len_mean: float = _TARGET_LEN_MEAN,
    target_len_std: float = _TARGET_LEN_STD,
    target_len_min: int = _TARGET_LEN_MIN,
    target_len_max: int = _TARGET_LEN_MAX,
    max_new_tokens: int = _MAX_NEW_TOKENS,
) -> list[RawSample]:
    """Transformers pipeline ile batch GPU inference yapar; A100 gibi büyük GPU'larda çok daha hızlı.

    target_len_*/max_new_tokens varsayilanlari ana 3000/sinif veri setinin (~1750 kelime
    ortalama, insan korpusuna hizali) parametreleridir. Kisa-metin pilot verisi (bkz.
    scripts/collect_short_pilot.py) gibi farkli bir hedef uzunluk gerektiren cagrilar
    bu degerleri override edebilir; varsayilan davranis (ana veri seti) degismez.
    """
    import json
    from dataclasses import asdict as _asdict

    pipe = _get_hf_pipeline(model_id, device, load_in_4bit)
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    items = [
        (
            start_index + j,
            prompts[(start_index + j) % len(prompts)],
            _sample_target_words(rng, target_len_mean, target_len_std, target_len_min, target_len_max),
        )
        for j in range(target_count)
    ]
    total = len(items)
    n_batches = (total + batch_size - 1) // batch_size
    samples: list[RawSample] = []

    for b in range(n_batches):
        batch_items = items[b * batch_size : (b + 1) * batch_size]
        messages_batch = [
            [{"role": "user", "content": _with_length_instruction(p, tw)}]
            for _, p, tw in batch_items
        ]

        print(f"  [batch {b+1}/{n_batches}] {len(batch_items)} prompt isleniyor...", flush=True)
        outputs = pipe(
            messages_batch,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=pipe.tokenizer.eos_token_id,
            batch_size=len(batch_items),
        )

        for k, (abs_i, prompt, target_words) in enumerate(batch_items):
            text = outputs[k][0]["generated_text"][-1]["content"].strip()
            if not text:
                print(f"  UYARI: bos yanit (indeks={abs_i}), atlaniyor.", flush=True)
                continue
            sample = RawSample(
                id=f"ai_raw_{provider}_{abs_i:04d}",
                text=text,
                label="ai_raw",
                source=provider,
                metadata={"prompt": prompt, "model": model_id, "target_words": target_words},
            )
            samples.append(sample)
            if checkpoint_path:
                with open(checkpoint_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(_asdict(sample), ensure_ascii=False) + "\n")

        done = min((b + 1) * batch_size, total)
        print(f"  [{done}/{total}] tamamlandi.", flush=True)

    return samples


def generate_batch(
    prompts: list[str],
    provider: str,
    target_count: int,
    rate_limit: dict | None = None,
    checkpoint_path: Path | None = None,
    start_index: int = 0,
    target_len_mean: float | None = None,
    target_len_std: float | None = None,
    target_len_min: int = _TARGET_LEN_MIN,
    target_len_max: int = _TARGET_LEN_MAX,
    max_concurrency: int = 1,
    **provider_kwargs,
) -> list[RawSample]:
    """Belirtilen saglayici ile toplu ham-AI metin uretir ve RawSample listesine cevirir.

    checkpoint_path verilirse her ornek uretilir uretilmez dosyaya eklenir (kesintide kayip olmaz).
    start_index ile onceki bir calismanin kaldigi yerden devam edilebilir.
    provider_kwargs, secilen saglayicinin generate_with_* fonksiyonuna gerekli
    diger parametreleri (model/api_key veya endpoint/api_key) tasir.

    target_len_mean/std verilirse (API saglayicilari icin, orn. openai/anthropic) her
    prompt'a _with_length_instruction ile rastgele bir hedef kelime sayisi enjekte edilir
    -- transformers yolundaki _sample_target_words mantigiyla ayni. Verilmezse (varsayilan,
    None) eski davranis korunur: prompt hicbir degisiklik olmadan gonderilir (bu saglayicilar
    tarihte hic gercek veri toplamada kullanilmadigi icin geriye donuk uyumluluk sorunu yok).

    max_concurrency>1 (API saglayicilari icin) istekleri ThreadPoolExecutor ile paralel
    gonderir -- bu is agindan/API gecikmesinden kaynaklanan (yerel GPU degil) darbogazi
    onemli olcude azaltir (bkz. proje notlari, 2026-07-19: olculen gercek hiz ~12sn/ornek
    sirali, GPU'nun hic devrede olmadigi bir API-cagrisi darbogazi -- Colab burada YARDIMCI
    OLMAZ, cunku ayni ag gecikmesiyle calisir). Varsayilan (1) eski sirali davranisi korur.
    """
    import json
    from dataclasses import asdict as _asdict

    if not prompts:
        raise ValueError("prompts listesi bos olamaz")

    # Transformers için doğrudan batch inference kullan (rate limit yok, GPU paralel).
    # "transformers_short" gibi bir alt-etiket de (kisa-pilot verisi icin, sample ID'lerini
    # ana veri setinden ayirt etmek amaciyla) ayni batch inference yolunu kullanir.
    if provider == "transformers" or provider.startswith("transformers_"):
        batch_size = int(provider_kwargs.pop("batch_size", 8))
        return _generate_batch_transformers(
            prompts, provider, target_count, start_index, checkpoint_path,
            batch_size=batch_size, **provider_kwargs,
        )

    generate_fn = _GENERATORS[provider]
    rate_limit = rate_limit or {}
    max_retries = rate_limit.get("max_retries", 3)
    requests_per_minute = rate_limit.get("requests_per_minute", 60)
    # Yerel sağlayıcılar (llama/ollama) için rate limit bekleme gereksiz
    limiter = _RateLimiter(None if provider in _LOCAL_PROVIDERS else requests_per_minute)

    retryer = Retrying(stop=stop_after_attempt(max_retries), wait=wait_exponential(multiplier=1, max=30))

    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def _build_prompt(i: int) -> str:
        prompt = prompts[i % len(prompts)]
        if target_len_mean is not None and target_len_std is not None:
            local_rng = random.Random(42 + i)
            target_words = _sample_target_words(local_rng, target_len_mean, target_len_std, target_len_min, target_len_max)
            prompt = _with_length_instruction(prompt, target_words)
        return prompt

    samples: list[RawSample] = []

    if max_concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        write_lock = threading.Lock()
        done_count = 0

        def _worker(i: int) -> tuple[int, str, str]:
            prompt = _build_prompt(i)
            limiter.acquire()
            try:
                text = retryer(generate_fn, prompt, **provider_kwargs).strip()
            except Exception as exc:
                print(f"  UYARI: indeks={i} icin kalici hata ({exc!r}), atlaniyor.", flush=True)
                text = ""
            return i, prompt, text

        indices = list(range(start_index, start_index + target_count))
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = {pool.submit(_worker, i): i for i in indices}
            for future in as_completed(futures):
                i = futures[future]
                _, prompt, text = future.result()
                with write_lock:
                    done_count += 1
                    if text:
                        sample = RawSample(
                            id=f"ai_raw_{provider}_{i:04d}",
                            text=text,
                            label="ai_raw",
                            source=provider,
                            metadata={"prompt": prompt, "model": provider_kwargs.get("model", "")},
                        )
                        samples.append(sample)
                        if checkpoint_path:
                            with open(checkpoint_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(_asdict(sample), ensure_ascii=False) + "\n")
                        print(f"  [{done_count}/{target_count}] tamam — id={sample.id}", flush=True)
                    else:
                        print(f"  [{done_count}/{target_count}] UYARI: bos yanit (indeks={i}), atlaniyor.", flush=True)
        return samples

    for i in range(start_index, start_index + target_count):
        local_i = i - start_index + 1
        print(f"  [{local_i}/{target_count}] {provider}: istek gonderiliyor (prompt #{i % len(prompts) + 1})...", flush=True)
        prompt = _build_prompt(i)
        limiter.acquire()
        try:
            text = retryer(generate_fn, prompt, **provider_kwargs).strip()
        except Exception as exc:
            print(f"  UYARI: indeks={i} icin kalici hata ({exc!r}), atlaniyor.", flush=True)
            text = ""
        if text:
            sample = RawSample(
                id=f"ai_raw_{provider}_{i:04d}",
                text=text,
                label="ai_raw",
                source=provider,
                metadata={"prompt": prompt, "model": provider_kwargs.get("model", "")},
            )
            samples.append(sample)
            if checkpoint_path:
                with open(checkpoint_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(_asdict(sample), ensure_ascii=False) + "\n")
            print(f"  [{local_i}/{target_count}] tamam — {len(text)} karakter yazildi (id={sample.id})", flush=True)
        else:
            print(f"  [{local_i}/{target_count}] UYARI: bos yanit, atlaniyor.", flush=True)
    return samples
