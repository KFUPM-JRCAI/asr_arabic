#!/bin/bash
# Test script for Qwen3-ASR-1.7B setup
set -e

echo "=== Testing Qwen3-ASR-1.7B Setup ==="
echo ""

# Check if docker-compose service starts
echo "1. Starting Qwen3-ASR vLLM service..."
docker compose --profile qwen3-asr up -d qwen3-asr

echo "2. Waiting for service to be ready (this may take a few minutes)..."
max_wait=300  # 5 minutes
waited=0
while [ $waited -lt $max_wait ]; do
    if curl -sf http://localhost:${QWEN3_ASR_HOST_PORT:-9012}/health > /dev/null 2>&1; then
        echo "   Service is ready!"
        break
    fi
    echo "   Waiting... ($waited/$max_wait seconds)"
    sleep 10
    waited=$((waited + 10))
done

if [ $waited -ge $max_wait ]; then
    echo "   ERROR: Service did not become ready in time"
    docker compose --profile qwen3-asr logs qwen3-asr
    exit 1
fi

echo ""
echo "3. Checking if models endpoint is available..."
curl -sf http://localhost:${QWEN3_ASR_HOST_PORT:-9012}/v1/models | jq .

echo ""
echo "4. Testing transcription with a small sample..."
# You can test with a local audio file or skip this step
echo "   (Skipping audio test - run evaluate.py for full test)"

echo ""
echo "=== Setup Complete ==="
echo "You can now run evaluation with:"
echo "docker compose run --rm leaderboard python scripts/evaluate.py \\"
echo "  --append \\"
echo "  --language ar \\"
echo "  --model Qwen/Qwen3-ASR-1.7B \\"
echo "  --api-url http://qwen3-asr:8000 \\"
echo "  --predictions-dir results/predictions_qwen3 \\"
echo "  --save-preds --resume"
