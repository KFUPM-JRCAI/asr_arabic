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

## MBZUAI/artst_asr_v3 (optional)

Run MBZUAI's ArTST-v3 Arabic ASR model (SpeechT5) via a local OpenAI-compatible wrapper.

```bash
# 1) Build the custom Docker image (first time only)
docker compose --profile artst-asr build artst-asr

# 2) Launch the service (GPU recommended)
docker compose --profile artst-asr up -d artst-asr

# 3) Monitor logs until the model is loaded
docker compose logs -f artst-asr
# Wait for: "Server ready!"

# 4) Evaluate against MBZUAI/artst_asr_v3
docker compose run --rm leaderboard python scripts/evaluate.py \
  --append \
  --language ar \
  --model MBZUAI/artst_asr_v3 \
  --api-url http://artst-asr:8099 \
  --predictions-dir results/predictions_artst_asr_v3 \
  --save-preds --resume
```

## MBZUAI/artst_asr_v3_qasr (optional)

Run MBZUAI's ArTST-v3 QASR Arabic ASR model (SpeechT5) via a local OpenAI-compatible wrapper.

```bash
# 1) Build the custom Docker image (first time only)
docker compose --profile artst-asr-qasr build artst-asr-qasr

# 2) Launch the service (GPU recommended)
docker compose --profile artst-asr-qasr up -d artst-asr-qasr

# 3) Monitor logs until the model is loaded
docker compose logs -f artst-asr-qasr
# Wait for: "Server ready!"

# 4) Evaluate against MBZUAI/artst_asr_v3_qasr
docker compose run --rm leaderboard python scripts/evaluate.py \
  --append \
  --language ar \
  --model MBZUAI/artst_asr_v3_qasr \
  --api-url http://artst-asr-qasr:8099 \
  --predictions-dir results/predictions_artst_asr_v3_qasr \
  --save-preds --resume
```

## MBZUAI/artst_asr_v2_qasr (optional)

Run MBZUAI's ArTST-v2 QASR Arabic ASR model (SpeechT5) via a local OpenAI-compatible wrapper.

```bash
# 1) Build the custom Docker image (first time only)
docker compose --profile artst-asr-v2-qasr build artst-asr-v2-qasr

# 2) Launch the service (GPU recommended)
docker compose --profile artst-asr-v2-qasr up -d artst-asr-v2-qasr

# 3) Monitor logs until the model is loaded
docker compose logs -f artst-asr-v2-qasr
# Wait for: "Server ready!"

# 4) Evaluate against MBZUAI/artst_asr_v2_qasr
docker compose run --rm leaderboard python scripts/evaluate.py \
  --append \
  --language ar \
  --model MBZUAI/artst_asr_v2_qasr \
  --api-url http://artst-asr-v2-qasr:8099 \
  --predictions-dir results/predictions_artst_asr_v2_qasr \
  --save-preds --resume
```

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
  --model KFUPM-JRCAI/WhisperTurboArabic_v2 \
  --api-url http://whisper-turbo-arabic:8099 \
  --predictions-dir results/predictions_whisper_turbo_arabic_v2 \
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

## facebook/omniASR-LLM-1B (optional)

Run Meta's omniASR-LLM-1B omnilingual ASR model (~6 GiB VRAM, BF16).

```bash
# 1) Build the custom Docker image (first time only)
docker compose --profile omniasr build omniasr

# 2) Launch the service (needs GPU)
docker compose --profile omniasr up -d omniasr

# 3) Monitor the logs until the model is loaded
docker compose logs -f omniasr
# Wait for: "Server ready!"

# 4) Evaluate against omniASR-LLM-1B
docker compose run --rm leaderboard python scripts/evaluate.py \
  --append \
  --language ar \
  --model facebook/omniASR-LLM-1B \
  --api-url http://omniasr:8099 \
  --predictions-dir results/predictions_omniasr \
  --save-preds --resume
```

## KFUPM-JRCAI/WhisperArabic_v4 (optional)

Run KFUPM-JRCAI's fine-tuned Whisper Large v3 model (v4) for Arabic ASR.
This is a CTranslate2/faster-whisper model, reusing the same Docker image as WhisperTurboArabic.

```bash
# 1) Build the Docker image (first time only, shared with WhisperTurboArabic)
docker compose --profile whisper-arabic-v4 build whisper-arabic-v4

# 2) Launch the service (needs GPU)
docker compose --profile whisper-arabic-v4 up -d whisper-arabic-v4

# 3) Monitor the logs until the model is loaded
docker compose logs -f whisper-arabic-v4
# Wait for: "Server ready!"

# 4) Evaluate against WhisperArabic_v4
docker compose run --rm leaderboard python scripts/evaluate.py \
  --append \
  --language ar \
  --model KFUPM-JRCAI/WhisperArabic_v4 \
  --api-url http://whisper-arabic-v4:8099 \
  --predictions-dir results/predictions_whisper_arabic_v4 \
  --save-preds --resume
```

## KFUPM-JRCAI/WhisperArabic_v3 (optional)

Run KFUPM-JRCAI's fine-tuned Whisper Large v3 model for Arabic ASR.
This is a CTranslate2/faster-whisper model, reusing the same Docker image as WhisperTurboArabic.

```bash
# 1) Build the Docker image (first time only, shared with WhisperTurboArabic)
docker compose --profile whisper-arabic-v3 build whisper-arabic-v3

# 2) Launch the service (needs GPU)
docker compose --profile whisper-arabic-v3 up -d whisper-arabic-v3

# 3) Monitor the logs until the model is loaded
docker compose logs -f whisper-arabic-v3
# Wait for: "Server ready!"

# 4) Evaluate against WhisperArabic_v3
docker compose run --rm leaderboard python scripts/evaluate.py \
  --append \
  --language ar \
  --model KFUPM-JRCAI/WhisperArabic_v3 \
  --api-url http://whisper-arabic-v3:8099 \
  --predictions-dir results/predictions_whisper_arabic_v3 \
  --save-preds --resume
```

## RaedMughaus/whisper-large-v3-finetuned-cv-corpus-24-ar (optional)

Run a LoRA-adapted Whisper Large v3 fine-tuned on Common Voice Corpus 24 (Arabic).
This loads the base `openai/whisper-large-v3` model and applies the PEFT/LoRA adapter.

```bash
# 1) Build the custom Docker image (first time only)
docker compose --profile whisper-lora-arabic build whisper-lora-arabic

# 2) Launch the service (needs GPU)
docker compose --profile whisper-lora-arabic up -d whisper-lora-arabic

# 3) Monitor the logs until the model is loaded
docker compose logs -f whisper-lora-arabic
# Wait for: "Server ready!"

# 4) Evaluate
docker compose run --rm leaderboard python scripts/evaluate.py \
  --append \
  --language ar \
  --model RaedMughaus/whisper-large-v3-finetuned-cv-corpus-24-ar \
  --api-url http://whisper-lora-arabic:8099 \
  --predictions-dir results/predictions_whisper_lora_arabic \
  --save-preds --resume
```

## UsefulSensors/moonshine-base-ar (optional)

Run Moonshine Base Arabic, a lightweight 61.5M parameter ASR model fine-tuned for Arabic by Useful Sensors.

```bash
# 1) Build the custom Docker image (first time only)
docker compose --profile moonshine build moonshine

# 2) Launch the service (needs GPU)
docker compose --profile moonshine up -d moonshine

# 3) Monitor the logs until the model is loaded
docker compose logs -f moonshine
# Wait for: "Server ready!"

# 4) Evaluate
docker compose run --rm leaderboard python scripts/evaluate.py \
  --append \
  --language ar \
  --model UsefulSensors/moonshine-base-ar \
  --api-url http://moonshine:8099 \
  --predictions-dir results/predictions_moonshine \
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
- `ARTST_ASR_HOST_PORT` (default 8087) - port for ArTST-v3 wrapper
- `ARTST_NUM_BEAMS` (default 10) - beam width used for ArTST-v3 generation
- `ARTST_MAX_LENGTH` (default 150) - max generation length used for ArTST-v3
- `ARTST_QASR_HOST_PORT` (default 8086) - port for ArTST-v3 QASR wrapper
- `ARTST_QASR_NUM_BEAMS` (default 10) - beam width used for ArTST-v3 QASR generation
- `ARTST_QASR_MAX_LENGTH` (default 150) - max generation length used for ArTST-v3 QASR
- `ARTST_V2_QASR_HOST_PORT` (default 8085) - port for ArTST-v2 QASR wrapper
- `ARTST_V2_QASR_NUM_BEAMS` (default 10) - beam width used for ArTST-v2 QASR generation
- `ARTST_V2_QASR_MAX_LENGTH` (default 150) - max generation length used for ArTST-v2 QASR
- `WHISPER_LARGE_ARABIC_HOST_PORT` (default 8096) - port for WhisperLargeArabic wrapper
- `WHISPER_TURBO_ARABIC_HOST_PORT` (default 8095) - port for WhisperTurboArabic wrapper
- `WHISPER_ARABIC_V3_HOST_PORT` (default 8093) - port for WhisperArabic_v3 wrapper
- `WHISPER_ARABIC_V4_HOST_PORT` (default 8092) - port for WhisperArabic_v4 wrapper
- `OMNIASR_HOST_PORT` (default 8094) - port for omniASR-LLM-1B wrapper
- `WHISPER_LORA_ARABIC_HOST_PORT` (default 8090) - port for whisper-large-v3 LoRA Arabic wrapper
