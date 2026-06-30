"""Asama 1: insan / ham-AI / humanized metinlerin toplanmasi.

Girdi : configs/data_sources.yaml + configs/paths.yaml + .env
Cikti : data/raw/{human,ai_raw,ai_humanized}/*.jsonl
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict

from humanai_detect.config import PROJECT_ROOT, get_api_key, load_yaml
from humanai_detect.data_collection import human_sources, humanizers, llm_generators
from humanai_detect.data_collection.schemas import RawSample
from humanai_detect.utils.io import read_jsonl, write_jsonl

_HUMAN_COLLECTORS = {
    "google_scholar": human_sources.collect_from_google_scholar,
    "yok_tez": human_sources.collect_from_yok_tez,
    "manual_corpus": human_sources.collect_from_manual_corpus,
}


def _split_target(total: int, n: int) -> int:
    return math.ceil(total / n) if n else 0


def collect_human(paths_cfg: dict, data_sources_cfg: dict) -> list[RawSample]:
    human_cfg = data_sources_cfg["human"]
    sources = human_cfg["sources"]
    per_source = _split_target(human_cfg["target_count"], len(sources))
    external_dir = PROJECT_ROOT / paths_cfg["external_dir"]

    samples: list[RawSample] = []
    for source in sources:
        subdir = human_cfg["external_subdirs"][source]
        collector = _HUMAN_COLLECTORS[source]
        samples.extend(collector(per_source, external_dir / subdir))
    return samples[: human_cfg["target_count"]]


def collect_ai_raw(paths_cfg: dict, data_sources_cfg: dict) -> list[RawSample]:
    out_path = PROJECT_ROOT / paths_cfg["raw_dir"] / "ai_raw" / "ai_raw.jsonl"
    target_count = data_sources_cfg["target_count_per_class"]

    # Resume: mevcut checkpoint'i yukle
    existing: list[RawSample] = []
    if out_path.exists():
        existing = [RawSample(**r) for r in read_jsonl(out_path)]
        if len(existing) >= target_count:
            print(f"[ai_raw] {len(existing)} ornek zaten mevcut, atlanıyor.")
            return existing[:target_count]
        print(f"[ai_raw] {len(existing)} mevcut ornek yuklendi, {target_count - len(existing)} daha uretilecek.")

    llm_cfg = data_sources_cfg["llm_generators"]
    rate_limit = data_sources_cfg.get("rate_limit")
    prompts = llm_generators.load_prompts(PROJECT_ROOT / llm_cfg["prompts_file"])

    providers = [name for name, cfg in llm_cfg.items() if name != "prompts_file" and cfg.get("enabled")]
    remaining = target_count - len(existing)
    per_provider = _split_target(remaining, len(providers)) if providers else 0

    new_samples: list[RawSample] = []
    for provider in providers:
        provider_cfg = llm_cfg[provider]
        if provider == "llama":
            kwargs = {
                "model": provider_cfg["model"],
                "endpoint": get_api_key(provider_cfg["endpoint_env"]),
                "api_key": get_api_key(provider_cfg["api_key_env"]),
            }
        elif provider == "transformers":
            kwargs = {
                "model_id": provider_cfg["model"],
                "device": provider_cfg.get("device", "auto"),
                "load_in_4bit": provider_cfg.get("load_in_4bit", True),
                "batch_size": provider_cfg.get("batch_size", 8),
            }
        else:
            kwargs = {
                "model": provider_cfg["model"],
                "api_key": get_api_key(provider_cfg["api_key_env"]),
            }
        new_samples.extend(
            llm_generators.generate_batch(
                prompts,
                provider,
                per_provider,
                rate_limit=rate_limit,
                checkpoint_path=out_path,
                start_index=len(existing),
                **kwargs,
            )
        )
    return (existing + new_samples)[:target_count]


def collect_ai_humanized(paths_cfg: dict, data_sources_cfg: dict) -> list[RawSample]:
    ai_raw_path = PROJECT_ROOT / paths_cfg["raw_dir"] / "ai_raw" / "ai_raw.jsonl"
    if not ai_raw_path.exists():
        raise FileNotFoundError(
            f"{ai_raw_path} bulunamadi. Once '--label ai_raw' ile ham-AI verisini toplayin."
        )
    ai_raw_samples = [RawSample(**record) for record in read_jsonl(ai_raw_path)]

    external_dir = PROJECT_ROOT / paths_cfg["external_dir"]
    humanizers_cfg = data_sources_cfg["humanizers"]
    rate_limit = data_sources_cfg.get("rate_limit")

    samples: list[RawSample] = []
    for tool, tool_cfg in humanizers_cfg.items():
        if tool == "llm":
            for provider, provider_cfg in tool_cfg.items():
                if not provider_cfg.get("enabled"):
                    continue
                # Checkpoint dosyasi: her ornek aninda yazilir, yeniden baslama destegi
                ckpt_path = PROJECT_ROOT / paths_cfg["raw_dir"] / "ai_humanized" / "ai_humanized.jsonl"
                existing: list[RawSample] = []
                if ckpt_path.exists():
                    existing = [RawSample(**r) for r in read_jsonl(ckpt_path)]
                start_index = len(existing)
                if start_index >= len(ai_raw_samples):
                    print(f"[ai_humanized/{provider}] {start_index} ornek zaten checkpoint'te — atlanıyor.")
                    samples.extend(existing)
                    continue
                if start_index:
                    print(f"[ai_humanized/{provider}] {start_index} ornek checkpoint'ten yuklendi, {len(ai_raw_samples) - start_index} kaldi.")
                if provider == "llama":
                    kwargs = {
                        "model": provider_cfg["model"],
                        "endpoint": get_api_key(provider_cfg["endpoint_env"]),
                        "api_key": get_api_key(provider_cfg["api_key_env"]),
                    }
                elif provider == "transformers":
                    kwargs = {
                        "model_id": provider_cfg["model"],
                        "device": provider_cfg.get("device", "auto"),
                        "load_in_4bit": provider_cfg.get("load_in_4bit", True),
                        "batch_size": provider_cfg.get("batch_size", 8),
                    }
                else:
                    kwargs = {
                        "model": provider_cfg["model"],
                        "api_key": get_api_key(provider_cfg["api_key_env"]),
                    }
                new_samples = humanizers.humanize_batch_llm(
                    ai_raw_samples,
                    provider,
                    rate_limit=rate_limit,
                    checkpoint_path=ckpt_path,
                    start_index=start_index,
                    **kwargs,
                )
                samples.extend(existing + new_samples)
            continue
        if not tool_cfg.get("enabled"):
            continue
        source_dir = external_dir / tool_cfg["external_subdir"]
        samples.extend(humanizers.humanize_batch(ai_raw_samples, tool, source_dir))
    return samples


def _write(label: str, samples: list[RawSample], paths_cfg: dict) -> None:
    out_path = PROJECT_ROOT / paths_cfg["raw_dir"] / label / f"{label}.jsonl"
    # ai_raw checkpoint ile aninda yazildi; toplam sayi eslesiyorsa tekrar yazma
    if label == "ai_raw" and out_path.exists():
        existing_count = sum(1 for _ in read_jsonl(out_path))
        if existing_count >= len(samples):
            print(f"[{label}] {existing_count} ornek checkpoint'ten zaten yazilmis -> {out_path}")
            return
    write_jsonl(out_path, (asdict(s) for s in samples))
    print(f"[{label}] {len(samples)} ornek yazildi -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        choices=["human", "ai_raw", "ai_humanized", "all"],
        default="all",
        help="Hangi sinif icin veri toplanacak",
    )
    args = parser.parse_args()

    paths_cfg = load_yaml("paths")
    data_sources_cfg = load_yaml("data_sources")

    labels = ["human", "ai_raw", "ai_humanized"] if args.label == "all" else [args.label]
    collectors = {
        "human": collect_human,
        "ai_raw": collect_ai_raw,
        "ai_humanized": collect_ai_humanized,
    }
    for label in labels:
        samples = collectors[label](paths_cfg, data_sources_cfg)
        _write(label, samples, paths_cfg)


if __name__ == "__main__":
    main()
