#!/usr/bin/env bash

# =============================================================================
# DefectFill — Slurm Batch Generation Job
# =============================================================================


ROOT="/mimer/NOBACKUP/groups/cast_fm/axel"
MODEL="$ROOT/Models/DefectGen1"
DATA="$ROOT/Data"

module purge
module load GCCcore/12.3.0
module load Python/3.11.3-GCCcore-12.3.0

source $ROOT/envs/defectgen_env/bin/activate   # adjust path if your venv lives elsewhere


# -----------------------------------------------------------------------------
# Force Python to flush output immediately.
# -----------------------------------------------------------------------------
export PYTHONUNBUFFERED=1


# -----------------------------------------------------------------------------
#  Go to your project directory (adjust path if different)
# -----------------------------------------------------------------------------
cd $MODEL


# -----------------------------------------------------------------------------
# Quick sanity check — confirm GPU is visible before launching training
# -----------------------------------------------------------------------------
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"


# -----------------------------------------------------------------------------
# Run training
# -----------------------------------------------------------------------------
python -u train.py \
    --data_root        $DATA/cwp_dataset \
    --defect_type      obj2 \
    --object_name      "rough textured dark grayscale metallic surface" \
    --output_dir       /mimer/NOBACKUP/groups/cast_fm/axel/Models/DefectGen1_output/output/cwp_uv_model_3\
    --pretrained_model_name sd2-community/stable-diffusion-2-inpainting \
    --train_steps      2500 \
    --save_steps       250 \
    --batch_size       4 \
    --unet_lr          2e-4 \
    --text_encoder_lr  4e-5 \
    --lora_rank        8 \
    --lora_alpha       16 \
    --lora_dropout     0.1 \
    --lambda_def       1.0 \
    --lambda_obj       0.1 \
    --lambda_attn      0.025 \
    --alpha            0.3 \
    --warmup_steps     100 \
    --mixed_precision  bf16 \
    --seed             42


