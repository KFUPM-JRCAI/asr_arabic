#!/usr/bin/env python3
"""Download HF-hosted Arabic ASR datasets and build JSONL manifests."""

from __future__ import annotations

import argparse
import io
import itertools
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import Audio, load_dataset
from datasets.utils.file_utils import xopen
import soundfile as sf
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
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Stream samples instead of downloading full splits "
            "(default: true for --smoke/--max-samples)"
        ),
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
    streaming: bool = False,
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
                streaming=streaming,
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


def iter_examples(dataset, max_samples: Optional[int]):
    if max_samples is None:
        return dataset, None
    if hasattr(dataset, "select"):
        try:
            dataset_len = len(dataset)
        except TypeError:
            dataset_len = None
        if dataset_len is not None:
            count = min(max_samples, dataset_len)
            return dataset.select(range(count)), count
    return itertools.islice(dataset, max_samples), max_samples


def manifest_rows(
    dataset,
    text_field: str,
    audio_field: str,
    base_dir: Path,
    max_samples: Optional[int],
):
    dataset = dataset.cast_column(audio_field, Audio(decode=False))
    iterator, total = iter_examples(dataset, max_samples)
    for example in tqdm(iterator, desc=base_dir.name, unit="utt", total=total):
        text = example.get(text_field, "")
        if not text:
            continue
        audio_path = audio_path_from_example(example, audio_field)
        yield {
            "audio_path": make_relative(audio_path, base_dir),
            "text": text,
        }


def sanitize_prefix(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def materialize_audio_rows(
    dataset,
    text_field: str,
    audio_field: str,
    base_dir: Path,
    max_samples: Optional[int],
    prefix: Optional[str] = None,
):
    dataset = dataset.cast_column(audio_field, Audio(decode=False))
    audio_dir = base_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    iterator, total = iter_examples(dataset, max_samples)
    safe_prefix = sanitize_prefix(prefix) if prefix else ""
    desc = safe_prefix or base_dir.name
    for idx, example in enumerate(
        tqdm(iterator, desc=desc, unit="utt", total=total)
    ):
        text = example.get(text_field, "")
        if not text:
            continue
        audio = example.get(audio_field)
        if isinstance(audio, dict):
            audio_bytes = audio.get("bytes")
            audio_path = audio.get("path")
            if audio_bytes is not None:
                with io.BytesIO(audio_bytes) as buf:
                    audio_array, sample_rate = sf.read(buf)
            elif audio_path:
                with xopen(str(audio_path), "rb") as f:
                    audio_array, sample_rate = sf.read(f)
            else:
                raise KeyError(f"Audio field '{audio_field}' missing path/bytes")
        elif isinstance(audio, str):
            with xopen(str(audio), "rb") as f:
                audio_array, sample_rate = sf.read(f)
        else:
            raise KeyError(f"Audio field '{audio_field}' missing data")
        name_prefix = f"{safe_prefix}_" if safe_prefix else ""
        output_path = audio_dir / f"{name_prefix}{idx:06d}.wav"
        sf.write(output_path, audio_array, sample_rate)
        yield {
            "audio_path": str(output_path.relative_to(base_dir)),
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
    streaming: bool,
):
    cache_dir = output_dir / spec["cache_id"] / "hf_cache"
    dataset = resolve_split(
        spec["hf_id"],
        split_preference,
        token,
        cache_dir,
        streaming=streaming,
    )
    output_key = spec["outputs"][0]
    dataset_dir = output_dir / output_key
    manifest_path = dataset_dir / "test.jsonl"
    if streaming:
        rows = materialize_audio_rows(
            dataset,
            spec["text_field"],
            spec["audio_field"],
            dataset_dir,
            max_samples,
        )
    else:
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
    streaming: bool,
):
    cache_dir = output_dir / spec["cache_id"] / "hf_cache"
    dataset = resolve_split(
        spec["hf_id"],
        split_preference,
        token,
        cache_dir,
        streaming=streaming,
    )
    dataset = dataset.cast_column(spec["audio_field"], Audio(decode=False))

    type_field = spec["type_field"]
    type_map = spec["type_map"]

    for type_value, output_key in type_map.items():
        dataset_dir = output_dir / output_key
        manifest_path = dataset_dir / "test.jsonl"
        filtered = dataset.filter(lambda row: row.get(type_field) == type_value)
        if streaming:
            rows = materialize_audio_rows(
                filtered,
                spec["text_field"],
                spec["audio_field"],
                dataset_dir,
                max_samples,
                prefix=output_key,
            )
        else:
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
    streaming: bool,
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
                streaming=streaming,
            )
            if streaming:
                rows = materialize_audio_rows(
                    dataset,
                    spec["text_field"],
                    spec["audio_field"],
                    dataset_dir,
                    max_samples,
                    prefix=config,
                )
            else:
                rows = manifest_rows(
                    dataset,
                    spec["text_field"],
                    spec["audio_field"],
                    dataset_dir,
                    max_samples,
                )
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                total += 1

    print(f"[casablanca] wrote {total} rows to {manifest_path}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    if args.smoke and args.max_samples is None:
        args.max_samples = 3
    if args.streaming is None:
        args.streaming = bool(args.smoke or args.max_samples)

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
                args.streaming,
            )
        elif key == "casablanca":
            process_casablanca(
                spec,
                split_preference,
                output_dir,
                args.overwrite,
                args.max_samples,
                args.hf_token,
                args.streaming,
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
                args.streaming,
            )

    return 0


def shutdown_fsspec() -> None:
    try:
        from fsspec import asyn
    except Exception:
        return
    loop = asyn.loop[0]
    thread = asyn.iothread[0]
    if loop is not None and loop.is_running():
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
    if thread is not None:
        thread.join(timeout=2)
    try:
        asyn.reset_lock()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        shutdown_fsspec()
