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

MODEL="cwp_uv_model_1"
STEP="1000"
CFG=1.8
# -----------------------------------------------------------------------------
# 7. After training — run generation
# -----------------------------------------------------------------------------
python -u generate.py \
    --pretrained_model_name  sd2-community/stable-diffusion-2-inpainting \
    --lora_weights_path      /mimer/NOBACKUP/groups/cast_fm/axel/Models/DefectGen1_output/output/$MODEL/unet_lora_step_${STEP} \
    --te_lora_weights_path   /mimer/NOBACKUP/groups/cast_fm/axel/Models/DefectGen1_output/output/$MODEL/text_encoder_lora_step_${STEP} \
    --good_images_dir        $DATA/cwp_dataset/test/good \
    --data_root              $DATA/cwp_dataset \
    --output_dir             /mimer/NOBACKUP/groups/cast_fm/axel/Data/Generated_data/obj2_uv/${MODEL}_ckpt${STEP}_cfg${CFG} \
    --object_name            "dark grayscale metallic surface" \
    --defect_type            obj2 \
    --num_inference_steps    50 \
    --guidance_scale         $CFG \
    --num_samples_lfs        4 \
    --mixed_precision        bf16 \
    --split_seed             42 \
    --train_fraction         0.0 \
    --seed                   42 \
    --mask_dilation_size     0 \
    --mask_blur_radius       5