#!/usr/bin/env python3
"""Gradio leaderboard UI for Arabic ASR evaluation results."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd

CONFIG_PATH = Path(os.getenv("DATASETS_CONFIG", "config/datasets.json"))
RESULTS_CSV = Path(os.getenv("LEADERBOARD_CSV", "results/leaderboard.csv"))


def load_config() -> list[dict]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f).get("datasets", [])


def expected_columns(datasets: list[dict]) -> list[str]:
    columns = ["Model", "Average WER", "Average CER"]
    for item in datasets:
        name = item["name"]
        columns.append(f"{name} WER")
        columns.append(f"{name} CER")
    return columns


def load_leaderboard() -> pd.DataFrame:
    datasets = load_config()
    columns = expected_columns(datasets)
    if not RESULTS_CSV.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(RESULTS_CSV)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[columns]
    if "Average WER" in df.columns:
        df = df.sort_values(by="Average WER", ascending=True, na_position="last")
    return df


def last_updated() -> str:
    if not RESULTS_CSV.exists():
        return "N/A"
    timestamp = datetime.fromtimestamp(RESULTS_CSV.stat().st_mtime)
    return timestamp.strftime("%Y-%m-%d %H:%M")


LEADERBOARD_TEXT = """
# Open Universal Arabic ASR Leaderboard (Local)

This local leaderboard mirrors the structure of the public Open Universal Arabic ASR Leaderboard, but
is scoped to the models you evaluate with the Speaches framework. It is designed to display results
for **Systran/faster-whisper-large-v3** (and any other models you evaluate).
"""

METRICS_TEXT = """
## Metrics

- **WER**: Word Error Rate (lower is better)
- **CER**: Character Error Rate (lower is better)

## Test Sets

- **SADA**: Saudi Audio Dataset
- **Common Voice 18.0**: Mozilla Common Voice Arabic
- **MASC (Clean)**: Modern Arabic Speech Corpus, clean split
- **MASC (Noisy)**: Modern Arabic Speech Corpus, noisy split
- **MGB-2**: Multi-Genre Broadcast (Arabic) challenge data
- **Casablanca**: Casablanca speech corpus

Metrics are computed with lightweight normalization (lowercasing, punctuation removal, and Arabic
 diacritics stripping). Adjust `scripts/evaluate.py` if you need exact parity with another benchmark.
"""

REQUEST_TEXT = """
## Request a model

Open an issue or share a model name if you want it added to the local leaderboard.
"""

CSS = """
#leaderboard-table {
  overflow-x: auto;
}
#leaderboard-table table {
  min-width: 1200px;
}
"""


with gr.Blocks(css=CSS) as demo:
    gr.Markdown(LEADERBOARD_TEXT)
    with gr.Tabs():
        with gr.TabItem("Leaderboard"):
            df = load_leaderboard()
            gr.Dataframe(
                value=df,
                interactive=False,
                elem_id="leaderboard-table",
                wrap=True,
            )
            gr.Markdown(f"Last updated: {last_updated()}")
        with gr.TabItem("Metrics"):
            gr.Markdown(METRICS_TEXT)
        with gr.TabItem("Request a model"):
            gr.Markdown(REQUEST_TEXT)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
