# Arabic ASR Leaderboard (local)

This repo scaffolds a local replica of the Open Universal Arabic ASR Leaderboard for
**Systran/faster-whisper-large-v3**, evaluated through the Speaches framework.

## Quick start (Docker)

1) Start Speaches (CPU image by default):

```bash
docker compose up -d speaches
```

2) Download the model inside Speaches:

```bash
curl -X POST http://localhost:8000/v1/models/Systran/faster-whisper-large-v3
```

3) Prepare datasets (see `datasets/README.md`) and edit `config/datasets.json` if needed.

4) Run evaluation (writes `results/leaderboard.csv`):

```bash
docker compose run --rm leaderboard python scripts/evaluate.py --append
```

5) Launch the leaderboard UI:

```bash
docker compose up leaderboard
```

Open `http://localhost:7860`.

## Quick start (local Python)

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/evaluate.py --append
python app.py
```

## Configuration

- `config/datasets.json`: dataset names + manifest paths
- `SPEACHES_BASE_URL`: Speaches API base URL (default `http://localhost:8000`)
- `MODEL_ID`: model name for evaluation (default `Systran/faster-whisper-large-v3`)
- `LEADERBOARD_CSV`: path to leaderboard CSV (default `results/leaderboard.csv`)

## Notes

- `scripts/evaluate.py` applies light text normalization (lowercase, punctuation removal,
  Arabic diacritics stripping). Adjust `normalize_text()` if you need parity with another
  benchmark.
- If you have a GPU, swap the Speaches image to a CUDA tag in `docker-compose.yml`.
