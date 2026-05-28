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


ACTIVATIONS_DIR="$ROOT/analysis/activations"
NUM_SAMPLES=1000

# ─── Run ───────────────────────────────────────────────────────────────────
cd "$ROOT/analysis"

$PYTHON scripts/cache_activations.py \
    --cache_n_examples "$NUM_SAMPLES" \
    --language_model "$MODEL" \
    --vision_model "$VISION" \
    --projector_weights "$PROJECTOR" \
    --output_dir "$ACTIVATIONS_DIR"

$PYTHON scripts/matching_rate.py \
    --scan-dir \
    --language-model "$MODEL" \
    --num_samples "$NUM_SAMPLES" \
    --num_freq_features "$NUM_SAMPLES" \
    --activations_dir "$ACTIVATIONS_DIR" \
    --output "$RESULTS_DIR/matching_rate.json" \
    --plot "$RESULTS_DIR/matching_rate.pdf"

$PYTHON scripts/reconstruction_sparsity.py \
    --num_samples "$NUM_SAMPLES" \
    --language-model "$MODEL" \
    --activations_dir "$ACTIVATIONS_DIR" \
    --plot "$RESULTS_DIR/reconstruction_sparsity.pdf"