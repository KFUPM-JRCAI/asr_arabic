# Arabic ASR Leaderboard (local)

Local replica of the Open Universal Arabic ASR Leaderboard for
**Systran/faster-whisper-large-v3**, evaluated through Speaches.

## Quick start (Docker)

```bash
docker compose up -d speaches
curl -X POST http://localhost:${SPEACHES_HOST_PORT:-8099}/v1/models/Systran/faster-whisper-large-v3

# full test splits
python scripts/download_datasets.py
# or smoke (3 samples each)
python scripts/download_datasets.py --smoke

# evaluate
nohup docker compose run --rm leaderboard python scripts/evaluate.py --append > results/eval.log 2>&1 &
```

Then open `http://localhost:${LEADERBOARD_HOST_PORT:-17860}`.

## Data format

Each dataset lives under `datasets/<dataset_id>/` with a `test.jsonl` manifest:

```json
{"audio_path": "audio/0001.wav", "text": "..."}
```

## Config knobs

- `SPEACHES_HOST_PORT` (default 8099)
- `LEADERBOARD_HOST_PORT` (default 17860)
- `HF_TOKEN` for gated datasets
- `SPEACHES_IMAGE` to swap CPU/GPU images
