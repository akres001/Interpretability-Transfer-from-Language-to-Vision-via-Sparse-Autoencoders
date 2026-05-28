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
VISION="dinov2"
PROJECTOR="$ROOT/train/weights/projector_model.pth"

ACTIVATIONS_LOC_DIR="$ROOT/analysis/activations_localization"
NUM_LOC_SAMPLES=380

# ─── Run ───────────────────────────────────────────────────────────────────
cd "$ROOT/analysis"

$PYTHON scripts/cache_activations.py \
    --language_model "$MODEL" \
    --vision_model "$VISION" \
    --use_images use_images_location.txt \
    --projector_weights "$PROJECTOR" \
    --output_dir "$ACTIVATIONS_LOC_DIR"

$PYTHON scripts/matching_rate.py \
    --scan-dir \
    --no-plot \
    --language-model "$MODEL" \
    --num_samples "$NUM_LOC_SAMPLES" \
    --num_freq_features "$NUM_LOC_SAMPLES" \
    --activations_dir "$ACTIVATIONS_LOC_DIR" \
    --output "$RESULTS_DIR/matching_rate_localization.json"

$PYTHON scripts/localization.py \
    --vision-model "$VISION" \
    --matching-results "$RESULTS_DIR/matching_rate_localization.json" \
    --output "$RESULTS_DIR/localization_results.json"