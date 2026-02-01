# Arabic ASR Leaderboard (local)

Local replica of the Open Universal Arabic ASR Leaderboard for
**Systran/faster-whisper-large-v3**, evaluated through Speaches.

## Quick start (Docker)

```bash
docker compose up -d speaches
curl -X POST http://localhost:${SPEACHES_HOST_PORT:-8099}/v1/models/Systran/faster-whisper-large-v3

# full test splits
python scripts/download_datasets.py
# or smoke (3 samples each, streamed to avoid full downloads)
python scripts/download_datasets.py --smoke
# force full downloads even with --smoke/--max-samples
python scripts/download_datasets.py --no-streaming --smoke

# evaluate
nohup docker compose run --rm leaderboard python scripts/evaluate.py --append > results/eval.log 2>&1 &
```

Then open `http://localhost:${LEADERBOARD_HOST_PORT:-17860}`.

## NVIDIA Riva / Parakeet (optional)

Run NVIDIA's Parakeet 1.1B RNNT multilingual NIM with the included OpenAI-compatible wrapper.

```bash
# 1) Launch the NIM (needs GPU + NGC_API_KEY)
export NGC_API_KEY=...
export NIM_TAGS_SELECTOR="mode=ofl,diarizer=disabled"
docker compose --profile riva up -d riva

# 2) Start the wrapper (maps ar -> ar-AR for Riva)
docker compose --profile riva up -d riva-wrapper

# 3) Evaluate against Riva
docker compose run --rm leaderboard python scripts/evaluate.py \
  --append \
  --language ar \
  --model parakeet-1-1b-rnnt-multilingual \
  --api-url http://riva-wrapper:8099 \
  --predictions-dir results/predictions_riva \
  --save-preds --resume
```

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
- `NGC_API_KEY` (required to pull NVIDIA NIM images when using `--profile riva`)
- `NIM_TAGS_SELECTOR` (e.g., `mode=ofl,diarizer=disabled` for offline Parakeet)
- `RIVA_WRAPPER_HOST_PORT` (default 8099)
- `RIVA_HTTP_HOST_PORT` / `RIVA_GRPC_HOST_PORT` to avoid host port clashes (default 9000/50051)
