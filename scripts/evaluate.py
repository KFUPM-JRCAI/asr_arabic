#!/usr/bin/env python3
"""Evaluate ASR via an OpenAI-compatible transcription API and update leaderboard CSV."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import httpx
import pandas as pd
from jiwer import cer, wer
from tqdm import tqdm

DEFAULT_CONFIG = "config/datasets.json"
DEFAULT_RESULTS = "results/leaderboard.csv"
DEFAULT_SPEACHES_URL = "http://localhost:8099"
DEFAULT_TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"
DEFAULT_MODEL = "Systran/faster-whisper-large-v3"


@dataclass
class DatasetConfig:
    dataset_id: str
    name: str
    path: Path


def load_config(path: Path) -> List[DatasetConfig]:
    config_dir = path.parent
    # Prefer repo root when config lives under config/ and datasets/ sits next to it.
    root_dir = config_dir
    if not (config_dir / "datasets").exists() and (config_dir.parent / "datasets").exists():
        root_dir = config_dir.parent
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    datasets = []
    for item in raw.get("datasets", []):
        dataset_path = Path(item["path"])
        if not dataset_path.is_absolute():
            dataset_path = (root_dir / dataset_path).resolve()
        datasets.append(
            DatasetConfig(
                dataset_id=item["id"],
                name=item["name"],
                path=dataset_path,
            )
        )
    return datasets


_AR_DIACRITICS_RE = re.compile(r"[\u064b-\u0652\u0670\u0640]")
_NON_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = _AR_DIACRITICS_RE.sub("", text)
    text = _NON_WORD_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def load_manifest(path: Path) -> Iterable[Tuple[Path, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    base_dir = path.parent
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            audio_path = Path(obj["audio_path"])
            if not audio_path.is_absolute():
                audio_path = base_dir / audio_path
            yield audio_path, obj["text"]


def transcribe(
    client: httpx.Client,
    audio_path: Path,
    model: str,
    language: str | None,
    prompt: str | None,
    response_format: str | None,
    transcriptions_path: str,
    timeout: float,
) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio file: {audio_path}")
    data = {"model": model}
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt
    if response_format:
        data["response_format"] = response_format
    with audio_path.open("rb") as f:
        files = {"file": (audio_path.name, f)}
        resp = client.post(
            transcriptions_path,
            data=data,
            files=files,
            timeout=timeout,
        )
    resp.raise_for_status()
    if response_format == "json":
        payload = resp.json()
        return payload.get("text", "")
    return resp.text.strip()


def compute_metrics(references: List[str], hypotheses: List[str]) -> Tuple[float, float]:
    return wer(references, hypotheses) * 100.0, cer(references, hypotheses) * 100.0


def evaluate_dataset(
    client: httpx.Client,
    dataset: DatasetConfig,
    model: str,
    language: str | None,
    prompt: str | None,
    response_format: str | None,
    transcriptions_path: str,
    timeout: float,
    max_samples: int | None,
    save_preds: Path | None,
    skip_errors: bool,
    resume: bool,
) -> Tuple[float, float]:
    references: List[str] = []
    hypotheses: List[str] = []
    preds_out = None
    pred_cache = {}

    if resume and save_preds and save_preds.exists():
        with save_preds.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                audio_path = obj.get("audio_path")
                if audio_path:
                    pred_cache[audio_path] = obj

    if save_preds:
        save_preds.parent.mkdir(parents=True, exist_ok=True)
        preds_out = save_preds.open("w", encoding="utf-8")

    try:
        samples_iter: Iterable[Tuple[Path, str]] = load_manifest(dataset.path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset '{dataset.name}' missing manifest at {dataset.path}"
        ) from exc

    if max_samples:
        samples_iter = (s for _, s in zip(range(max_samples), samples_iter))

    for audio_path, ref_text in tqdm(
        samples_iter, desc=dataset.name, unit="file", total=max_samples
    ):
        cache_key = str(audio_path)
        try:
            if resume and cache_key in pred_cache:
                hyp_text = pred_cache[cache_key].get("prediction", "")
            else:
                hyp_text = transcribe(
                    client,
                    audio_path,
                    model=model,
                    language=language,
                    prompt=prompt,
                    response_format=response_format,
                    transcriptions_path=transcriptions_path,
                    timeout=timeout,
                )
        except Exception as exc:  # noqa: BLE001 - CLI tool should optionally continue
            if skip_errors:
                print(f"[WARN] {dataset.name}: {audio_path} failed: {exc}", file=sys.stderr)
                continue
            raise
        norm_ref = normalize_text(ref_text)
        norm_hyp = normalize_text(hyp_text)
        references.append(norm_ref)
        hypotheses.append(norm_hyp)
        if preds_out:
            preds_out.write(
                json.dumps(
                    {
                        "audio_path": str(audio_path),
                        "reference": ref_text,
                        "prediction": hyp_text,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    if preds_out:
        preds_out.close()

    if not references:
        raise RuntimeError(f"No samples processed for dataset '{dataset.name}'")

    return compute_metrics(references, hypotheses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Dataset config JSON")
    parser.add_argument(
        "--api-url",
        default=os.getenv("ASR_API_BASE_URL", os.getenv("SPEACHES_BASE_URL", DEFAULT_SPEACHES_URL)),
        help="Base URL for the transcription API (OpenAI-compatible).",
    )
    parser.add_argument(
        "--speaches-url",
        dest="api_url",
        help="Deprecated alias for --api-url (kept for compatibility).",
    )
    parser.add_argument("--model", default=os.getenv("MODEL_ID", DEFAULT_MODEL))
    parser.add_argument("--language", default="ar")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--response-format", default="text", choices=["text", "json"])
    parser.add_argument(
        "--transcriptions-path",
        default=DEFAULT_TRANSCRIPTIONS_PATH,
        help="Path for the transcriptions endpoint.",
    )
    parser.add_argument(
        "--auth-token",
        default=os.getenv("ASR_API_TOKEN", os.getenv("SPEACHES_API_TOKEN")),
        help="Bearer token for Authorization header (or set ASR_API_TOKEN).",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--append", action="store_true", help="Append/replace model row in leaderboard")
    parser.add_argument("--save-preds", action="store_true", help="Save per-utterance predictions")
    parser.add_argument("--resume", action="store_true", help="Reuse saved predictions to skip already-processed files (implies --save-preds)")
    parser.add_argument(
        "--predictions-dir",
        default="results/predictions",
        help="Directory for per-utterance prediction JSONLs (default: results/predictions)",
    )
    parser.add_argument("--skip-errors", action="store_true", help="Skip files that fail to transcribe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume and not args.save_preds:
        args.save_preds = True
    config_path = Path(args.config)
    results_path = Path(args.results)
    datasets = load_config(config_path)

    results_path.parent.mkdir(parents=True, exist_ok=True)

    row = {"Model": args.model}
    wer_scores: List[float] = []
    cer_scores: List[float] = []

    headers = {}
    if args.auth_token:
        headers["Authorization"] = f"Bearer {args.auth_token}"

    with httpx.Client(base_url=args.api_url, headers=headers) as client:
        for dataset in datasets:
            pred_path = None
            if args.save_preds:
                pred_path = Path(args.predictions_dir) / f"{dataset.dataset_id}.jsonl"
            wer_val, cer_val = evaluate_dataset(
                client,
                dataset,
                model=args.model,
                language=args.language,
                prompt=args.prompt,
                response_format=args.response_format,
                transcriptions_path=args.transcriptions_path,
                timeout=args.timeout,
                max_samples=args.max_samples,
                save_preds=pred_path,
                skip_errors=args.skip_errors,
                resume=args.resume,
            )
            row[f"{dataset.name} WER"] = round(wer_val, 2)
            row[f"{dataset.name} CER"] = round(cer_val, 2)
            wer_scores.append(wer_val)
            cer_scores.append(cer_val)

    row["Average WER"] = round(sum(wer_scores) / max(len(wer_scores), 1), 2)
    row["Average CER"] = round(sum(cer_scores) / max(len(cer_scores), 1), 2)

    if results_path.exists() and args.append:
        existing = pd.read_csv(results_path)
        existing = existing[existing["Model"] != args.model]
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])

    updated.to_csv(results_path, index=False)
    print(f"Saved leaderboard to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
