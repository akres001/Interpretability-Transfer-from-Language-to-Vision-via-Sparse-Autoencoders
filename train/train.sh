#!/bin/bash

export WANDB_API_KEY=""
export HF_TOKEN=""

MODEL="gemma-2-2b-it"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"

SAE_LAYERS="0 1 2 3 4"
SAE_LAYER_STR=""
VISION_MODEL='dinov2'

# This will count the number of items in SAE_LAYERS.
if [ -n "$SAE_LAYERS" ]; then
    SAE_LAYER_STR=$(echo $SAE_LAYERS | tr ' ' '-')
    SAE_LAYERS_ARRAY=($SAE_LAYERS)
    NSAES=${#SAE_LAYERS_ARRAY[@]}
    SAE_ARGS="--sae_layers $SAE_LAYERS"
else
    NSAES=0
    SAE_LAYER_STR="none" # Or any placeholder like "none"
    SAE_ARGS="--nsaes $NSAES --sae_layer_str $SAE_LAYER_STR"
fi

echo $NSAES


PRETRAIN_ARGS="
    --gradient_accumulation_steps 64 \
    --max_length 512 \
    --batch_size 1 \
    --n_batches 100000000 \
    --n_epochs 1 \
    --save_every 100 \
    --initial_learning_rate 1e-3 \
    --min_learning_rate 1e-8 \
    --warmup_ratio 0.03 \
    --name pretrain \
    --json_file /app/vlm_sae/vlm-mapping/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_dir /app/vlm_sae/vlm-mapping/data/LLaVA-Pretrain/images/ \
    --language_model $MODEL \
    --d_type bfloat16 \
    --sae_constraints 1 \
    --vision_model $VISION_MODEL
    $SAE_ARGS "


# Define finetune config
FINETUNE_ARGS="
    --gradient_accumulation_steps 32 \
    --max_length 1024 \
    --batch_size 1 \
    --n_batches 1000000000 \
    --save_every 100 \
    --n_epochs 3 \
    --initial_learning_rate 2e-5 \
    --min_learning_rate 1e-8 \
    --warmup_ratio 0 \
    --name finetune \
    --json_file /app/vlm_sae/vlm-mapping/data/LLaVA-Instruct/llava_v1_5_mix665k.json \
    --image_dir /app/vlm_sae/vlm-mapping/data/LLaVA-Instruct/ \
    --language_model $MODEL \
    --d_type bfloat16 \
    --sae_constraints 1 \
    --vision_model $VISION_MODEL \
    $SAE_ARGS "



echo "Starting pretraining..."
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port=29510 --config_file default_config.yaml train.py $PRETRAIN_ARGS


# # Extract language model name and training name from arguments
LM_NAME=$(echo "$PRETRAIN_ARGS" | grep -o "language_model [^ ]*" | cut -d' ' -f2)
VM_NAME=$(echo "$PRETRAIN_ARGS" | grep -o "vision_model [^ ]*" | cut -d' ' -f2)
PRETRAIN_NAME=$(echo "$PRETRAIN_ARGS" | grep -o "name [^ ]*" | cut -d' ' -f2)
FINETUNE_NAME=$(echo "$FINETUNE_ARGS" | grep -o "name [^ ]*" | cut -d' ' -f2)

SAFE_LM_NAME="${LM_NAME//\//_}"

cp "/app/vlm_sae/vlm-mapping/train/weights/projector_${SAFE_LM_NAME}_${VM_NAME}_${PRETRAIN_NAME}_saes_${SAE_LAYER_STR}.pth" \
   "/app/vlm_sae/vlm-mapping/train/weights/projector_${SAFE_LM_NAME}_${VM_NAME}_${FINETUNE_NAME}_saes_${SAE_LAYER_STR}.pth"
sleep 20

echo "Starting finetuning..."
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port=29510 --config_file default_config.yaml train.py $FINETUNE_ARGS


echo "All training completed!"