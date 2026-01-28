#!/usr/bin/env python3
"""Download HF-hosted Arabic ASR datasets and build JSONL manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import Audio, load_dataset
from tqdm import tqdm


DATASET_SPECS = {
    "sada": {
        "hf_id": "MohamedRashad/SADA22",
        "text_field": "text",
        "audio_field": "audio",
        "cache_id": "sada",
        "outputs": ["sada"],
        "fallback_splits": ["validation", "dev"],
    },
    "common_voice_18": {
        "hf_id": "MohamedRashad/common-voice-18-arabic",
        "text_field": "sentence",
        "audio_field": "audio",
        "cache_id": "common_voice_18",
        "outputs": ["common_voice_18"],
        "fallback_splits": ["validation"],
    },
    "masc": {
        "hf_id": "MohamedRashad/MASC-Arabic",
        "text_field": "text",
        "audio_field": "audio",
        "type_field": "type",
        "type_map": {"c": "masc_clean", "n": "masc_noisy"},
        "cache_id": "masc",
        "outputs": ["masc_clean", "masc_noisy"],
        "fallback_splits": ["validation"],
    },
    "mgb2": {
        "hf_id": "MohamedRashad/mgb2-arabic",
        "text_field": "transcript",
        "audio_field": "audio",
        "cache_id": "mgb2",
        "outputs": ["mgb2"],
        "fallback_splits": ["validation", "dev"],
    },
    "casablanca": {
        "hf_id": "UBC-NLP/Casablanca",
        "text_field": "transcription",
        "audio_field": "audio",
        "configs": [
            "Algeria",
            "Egypt",
            "Jordan",
            "Mauritania",
            "Morocco",
            "Palestine",
            "UAE",
            "Yemen",
        ],
        "cache_id": "casablanca",
        "outputs": ["casablanca"],
        "fallback_splits": ["validation"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated dataset keys (default: all)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Download a tiny subset (default 3 samples per dataset)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Primary split to download (default: test)",
    )
    parser.add_argument(
        "--allow-validation",
        action="store_true",
        help="If split is missing, fall back to validation/dev (still no train)",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets",
        help="Root directory for dataset caches/manifests",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on samples per dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing manifests",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN", None),
        help="Hugging Face token for gated datasets (or set HF_TOKEN)",
    )
    return parser.parse_args()


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {path} (use --overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict], overwrite: bool) -> int:
    ensure_output_path(path, overwrite)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def resolve_split(
    hf_id: str,
    split_preference: List[str],
    token: Optional[str],
    cache_dir: Path,
    config: Optional[str] = None,
):
    last_error = None
    for split in split_preference:
        try:
            return load_dataset(
                hf_id,
                name=config,
                split=split,
                cache_dir=str(cache_dir),
                token=token,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    raise RuntimeError(
        f"Could not load splits {split_preference} for {hf_id}"
    ) from last_error


def audio_path_from_example(example: dict, audio_field: str) -> Path:
    audio = example.get(audio_field)
    if isinstance(audio, dict) and "path" in audio:
        return Path(audio["path"])
    if isinstance(audio, str):
        return Path(audio)
    raise KeyError(f"Audio field '{audio_field}' missing path")


def make_relative(path: Path, base_dir: Path) -> str:
    if path.is_absolute():
        try:
            return str(path.relative_to(base_dir))
        except ValueError:
            return str(path)
    return str(path)


def manifest_rows(
    dataset,
    text_field: str,
    audio_field: str,
    base_dir: Path,
    max_samples: Optional[int],
):
    dataset = dataset.cast_column(audio_field, Audio(decode=False))
    iterator = dataset
    if max_samples:
        iterator = dataset.select(range(min(max_samples, len(dataset))))
    for example in tqdm(iterator, desc=base_dir.name, unit="utt"):
        text = example.get(text_field, "")
        if not text:
            continue
        audio_path = audio_path_from_example(example, audio_field)
        yield {
            "audio_path": make_relative(audio_path, base_dir),
            "text": text,
        }


def process_simple_dataset(
    key: str,
    spec: dict,
    split_preference: List[str],
    output_dir: Path,
    overwrite: bool,
    max_samples: Optional[int],
    token: Optional[str],
):
    cache_dir = output_dir / spec["cache_id"] / "hf_cache"
    dataset = resolve_split(
        spec["hf_id"],
        split_preference,
        token,
        cache_dir,
    )
    output_key = spec["outputs"][0]
    dataset_dir = output_dir / output_key
    manifest_path = dataset_dir / "test.jsonl"
    rows = manifest_rows(
        dataset,
        spec["text_field"],
        spec["audio_field"],
        dataset_dir,
        max_samples,
    )
    count = write_jsonl(manifest_path, rows, overwrite)
    print(f"[{key}] wrote {count} rows to {manifest_path}")


def process_masc(
    spec: dict,
    split_preference: List[str],
    output_dir: Path,
    overwrite: bool,
    max_samples: Optional[int],
    token: Optional[str],
):
    cache_dir = output_dir / spec["cache_id"] / "hf_cache"
    dataset = resolve_split(
        spec["hf_id"],
        split_preference,
        token,
        cache_dir,
    )
    dataset = dataset.cast_column(spec["audio_field"], Audio(decode=False))

    type_field = spec["type_field"]
    type_map = spec["type_map"]

    for type_value, output_key in type_map.items():
        dataset_dir = output_dir / output_key
        manifest_path = dataset_dir / "test.jsonl"
        filtered = dataset.filter(lambda row: row.get(type_field) == type_value)
        rows = manifest_rows(
            filtered,
            spec["text_field"],
            spec["audio_field"],
            dataset_dir,
            max_samples,
        )
        count = write_jsonl(manifest_path, rows, overwrite)
        print(f"[masc:{output_key}] wrote {count} rows to {manifest_path}")


def process_casablanca(
    spec: dict,
    split_preference: List[str],
    output_dir: Path,
    overwrite: bool,
    max_samples: Optional[int],
    token: Optional[str],
):
    cache_dir = output_dir / spec["cache_id"] / "hf_cache"
    dataset_dir = output_dir / spec["outputs"][0]
    manifest_path = dataset_dir / "test.jsonl"
    ensure_output_path(manifest_path, overwrite)

    total = 0
    with manifest_path.open("w", encoding="utf-8") as f:
        for config in spec["configs"]:
            dataset = resolve_split(
                spec["hf_id"],
                split_preference,
                token,
                cache_dir,
                config=config,
            )
            dataset = dataset.cast_column(spec["audio_field"], Audio(decode=False))
            iterator = dataset
            if max_samples:
                iterator = dataset.select(range(min(max_samples, len(dataset))))
            for example in tqdm(iterator, desc=f"casablanca:{config}", unit="utt"):
                text = example.get(spec["text_field"], "")
                if not text:
                    continue
                audio_path = audio_path_from_example(example, spec["audio_field"])
                row = {
                    "audio_path": make_relative(audio_path, dataset_dir),
                    "text": text,
                }
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                total += 1

    print(f"[casablanca] wrote {total} rows to {manifest_path}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    if args.smoke and args.max_samples is None:
        args.max_samples = 3

    if args.only:
        selected = {item.strip() for item in args.only.split(",") if item.strip()}
    else:
        selected = set(DATASET_SPECS.keys())

    for key in sorted(selected):
        if key not in DATASET_SPECS:
            raise KeyError(f"Unknown dataset key: {key}")

        spec = DATASET_SPECS[key]
        split_preference = [args.split]
        if args.allow_validation:
            split_preference.extend(spec.get("fallback_splits", []))

        if key == "masc":
            process_masc(
                spec,
                split_preference,
                output_dir,
                args.overwrite,
                args.max_samples,
                args.hf_token,
            )
        elif key == "casablanca":
            process_casablanca(
                spec,
                split_preference,
                output_dir,
                args.overwrite,
                args.max_samples,
                args.hf_token,
            )
        else:
            process_simple_dataset(
                key,
                spec,
                split_preference,
                output_dir,
                args.overwrite,
                args.max_samples,
                args.hf_token,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
