#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python}"


# ─── Secrets & paths ───────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1
export HF_TOKEN="hf_..."
export NEURONPEDIA_API_KEY="sk-np-..."
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/vertexkey.json"

ROOT="$(cd "$(dirname "$0")" && pwd)"
export VLM_SAE_ROOT="$ROOT"
export VLM_SAE_DATA="$ROOT/data"
export COCO_PATH="$ROOT/data/LLaVA-Instruct/coco"
export LLAVA_JSON="$ROOT/data/LLaVA-Instruct/llava_v1_5_mix665k.json"
export RESULTS_DIR="$ROOT/results"
mkdir -p "$RESULTS_DIR"

# ─── Config ────────────────────────────────────────────────────────────────
MODEL="gemma-2-2b-it"
SAFE_MODEL="${MODEL//\//_}"
VISION="dinov2"
PROJECTOR="$ROOT/train/weights/projector_model.pth"


# ─── Run ───────────────────────────────────────────────────────────────────
cd "$ROOT/analysis"

$PYTHON scripts/steering.py \
    --language-model "$MODEL" \
    --vision-model "$VISION" \
    --projector-weights "$PROJECTOR" \
    --images-file steering_imgs.py \
    --output "$RESULTS_DIR/steering_${SAFE_MODEL}_${VISION}.json"