#!/usr/bin/env bash
# =============================================================================
# DefectFill — Alvis Interactive Session Commands
# Account: NAISS2026-4-280  |  GPU: A40  |  Env: defectgen_env
# =============================================================================
# Paste these blocks one at a time into your terminal.
# =============================================================================


# -----------------------------------------------------------------------------
# 1. Request an interactive GPU node (if not already in one)
#    Skip this if squeue -u $USER already shows your session running.
# -----------------------------------------------------------------------------
#salloc \
#    --account=NAISS2026-4-280 \
#    --gpus-per-node=A40:1 \
#    --ntasks-per-node=4 \
#    --time=02:00:00 \
#    --mem=32G

ROOT="/mimer/NOBACKUP/groups/cast_fm/axel"
MODEL="$ROOT/Models/DefectGen1"
DATA="$ROOT/Data"


# -----------------------------------------------------------------------------
# 2. Once the shell opens on the compute node — activate your environment
# -----------------------------------------------------------------------------
source $ROOT/envs/defectgen_env/bin/activate   # adjust path if your venv lives elsewhere


# -----------------------------------------------------------------------------
# 3. Fix: force Python to flush output immediately.
#    This is why you see nothing — Python buffers stdout when not in a TTY.
# -----------------------------------------------------------------------------
export PYTHONUNBUFFERED=1


# -----------------------------------------------------------------------------
# 4. Go to your project directory (adjust path if different)
# -----------------------------------------------------------------------------
cd $MODEL


# -----------------------------------------------------------------------------
# 5. Quick sanity check — confirm GPU is visible before launching training
# -----------------------------------------------------------------------------
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"


# -----------------------------------------------------------------------------
# 6. Run training
#    -u flag = unbuffered (belt-and-suspenders alongside PYTHONUNBUFFERED)
# -----------------------------------------------------------------------------
python -u train.py \
    --data_root        $DATA/cwp_dataset \
    --defect_type      obj2_uv \
    --object_name      "rough textured dark grayscale metallic surface" \
    --output_dir       /mimer/NOBACKUP/groups/cast_fm/axel/Models/DefectGen1_output/output/cwp_uv_model_1 \
    --pretrained_model_name sd2-community/stable-diffusion-2-inpainting \
    --train_steps      1500 \
    --save_steps       250 \
    --batch_size       4 \
    --unet_lr          2e-4 \
    --text_encoder_lr  4e-5 \
    --lora_rank        8 \
    --lora_alpha       16 \
    --lora_dropout     0.1 \
    --lambda_def       0.5 \
    --lambda_obj       0.2 \
    --lambda_attn      0.05 \
    --alpha            0.3 \
    --warmup_steps     100 \
    --mixed_precision  bf16 \
    --seed             42


