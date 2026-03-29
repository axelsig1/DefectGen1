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
# force Python to flush output immediately.
# -----------------------------------------------------------------------------
export PYTHONUNBUFFERED=1


# -----------------------------------------------------------------------------
# Go to project directory
# -----------------------------------------------------------------------------
cd $MODEL

# Define absolute path to venv Python
VENV_PYTHON="$ROOT/envs/defectgen_env/bin/python"

# -----------------------------------------------------------------------------
# Quick sanity check — confirm GPU is visible before launching inference
# -----------------------------------------------------------------------------
$VENV_PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

MODEL="cwp_uv_model_2"
STEP="1750"
CFG=2.0
# -----------------------------------------------------------------------------
# Run generation
# -----------------------------------------------------------------------------
$VENV_PYTHON -u generate.py \
    --pretrained_model_name  sd2-community/stable-diffusion-2-inpainting \
    --lora_weights_path      /mimer/NOBACKUP/groups/cast_fm/axel/Models/DefectGen1_output/output/$MODEL/unet_lora_step_${STEP} \
    --te_lora_weights_path   /mimer/NOBACKUP/groups/cast_fm/axel/Models/DefectGen1_output/output/$MODEL/text_encoder_lora_step_${STEP} \
    --good_images_dir        $DATA/cwp_dataset/test/good \
    --data_root              $DATA/cwp_dataset \
    --output_dir             /mimer/NOBACKUP/groups/cast_fm/axel/Data/Generated_data/obj2_uv/${MODEL}_ckpt${STEP}_cfg${CFG} \
    --object_name            "rough textured dark grayscale metallic surface" \
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