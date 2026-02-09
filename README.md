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

## NVIDIA Canary-1B (optional)

Run NVIDIA's Canary-1B ASR NIM (supports both speech-to-text recognition and translation).

```bash
# 1) Launch the Canary NIM (needs GPU + NGC_API_KEY)
export NGC_API_KEY=...
docker compose --profile canary up -d canary

# 2) Start the wrapper (maps ar -> ar-AR for Canary)
docker compose --profile canary up -d canary-wrapper

# 3) Evaluate against Canary
docker compose run --rm leaderboard python scripts/evaluate.py \
  --append \
  --language ar \
  --model canary-1b \
  --api-url http://canary-wrapper:8099 \
  --predictions-dir results/predictions_canary \
  --save-preds --resume
```

## Qwen3-ASR-1.7B (optional)

Run Qwen3-ASR-1.7B via the official `qwen-asr` package with vLLM backend.

```bash
# 1) Build the custom Docker image (first time only, may take 10-15 minutes)
docker compose --profile qwen3-asr build qwen3-asr

# 2) Launch the vLLM service and wrapper (needs GPU)
docker compose --profile qwen3-asr up -d qwen3-asr qwen3-asr-wrapper

# 3) Monitor the logs until the model is loaded and server is ready (may take 5-10 minutes)
docker compose logs -f qwen3-asr
# Wait for: "Uvicorn running on http://0.0.0.0:8000"

# 4) Evaluate against Qwen3-ASR via the wrapper
docker compose run --rm leaderboard python scripts/evaluate.py \
  --append \
  --language ar \
  --model Qwen/Qwen3-ASR-1.7B \
  --api-url http://qwen3-asr-wrapper:8099 \
  --predictions-dir results/predictions_qwen3 \
  --save-preds --resume
```

**Note:** This uses the official `qwen-asr-serve` command which includes the necessary transformers updates to support the new `qwen3_asr` model architecture. The wrapper handles the JSON response format and forces Arabic language detection.

## KFUPM-JRCAI/WhisperTurboArabic (optional)

Run KFUPM-JRCAI's fine-tuned Whisper Large v3 Turbo model for Arabic ASR.
This is a CTranslate2/faster-whisper model served via a custom wrapper (not in the Speaches registry).

```bash
# 1) Build the custom Docker image (first time only)
docker compose --profile whisper-turbo-arabic build whisper-turbo-arabic

# 2) Launch the service (needs GPU)
docker compose --profile whisper-turbo-arabic up -d whisper-turbo-arabic

# 3) Monitor the logs until the model is loaded
docker compose logs -f whisper-turbo-arabic
# Wait for: "Server ready!"

# 4) Evaluate against WhisperTurboArabic
docker compose run --rm leaderboard python scripts/evaluate.py \
  --append \
  --language ar \
  --model KFUPM-JRCAI/WhisperTurboArabic \
  --api-url http://whisper-turbo-arabic:8099 \
  --predictions-dir results/predictions_whisper_turbo_arabic \
  --save-preds --resume
```

## KFUPM-JRCAI/WhisperLargeArabic (optional)

Run KFUPM-JRCAI's fine-tuned Whisper Large model for Arabic ASR.

```bash
# 1) Build the custom Docker image (first time only)
docker compose --profile whisper-large-arabic build whisper-large-arabic

# 2) Launch the service (needs GPU)
docker compose --profile whisper-large-arabic up -d whisper-large-arabic

# 3) Monitor the logs until the model is loaded
docker compose logs -f whisper-large-arabic
# Wait for: "Server ready!"

# 4) Evaluate against WhisperLargeArabic
docker compose run --rm leaderboard python scripts/evaluate.py \
  --append \
  --language ar \
  --model KFUPM-JRCAI/WhisperLargeArabic \
  --api-url http://whisper-large-arabic:8099 \
  --predictions-dir results/predictions_whisper_large_arabic \
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
- `NGC_API_KEY` (required to pull NVIDIA NIM images when using `--profile riva` or `--profile canary`)
- `NIM_TAGS_SELECTOR` (e.g., `mode=ofl,diarizer=disabled` for offline Parakeet, or `name=canary-1b` for Canary)
- `RIVA_WRAPPER_HOST_PORT` (default 8099) - wrapper port for Parakeet
- `RIVA_HTTP_HOST_PORT` / `RIVA_GRPC_HOST_PORT` - Parakeet NIM ports (default 9000/50051)
- `CANARY_WRAPPER_HOST_PORT` (default 8098) - wrapper port for Canary
- `CANARY_HTTP_HOST_PORT` / `CANARY_GRPC_HOST_PORT` - Canary NIM ports (default 9011/50052)
- `QWEN3_WRAPPER_HOST_PORT` (default 8097) - wrapper port for Qwen3-ASR
- `QWEN3_ASR_HOST_PORT` (default 9012) - vLLM server port for Qwen3-ASR
- `QWEN3_GPU_MEMORY_UTIL` (default 0.8) - GPU memory utilization for Qwen3-ASR vLLM
- `QWEN3_MAX_MODEL_LEN` (default 4096) - max model length for Qwen3-ASR vLLM
- `WHISPER_LARGE_ARABIC_HOST_PORT` (default 8096) - port for WhisperLargeArabic wrapper
- `WHISPER_TURBO_ARABIC_HOST_PORT` (default 8095) - port for WhisperTurboArabic wrapper
