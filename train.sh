#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-280
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 0-02:00:00              # Set to 2 hours based on your interactive salloc request
#SBATCH -J defectfill_train
#SBATCH --output=/mimer/NOBACKUP/groups/cast_fm/axel/Models/DefectGen1_output/logs/test_fix_defectfill_1_train_%j.out


# =============================================================================
# DefectFill — Slurm Batch Train Job
# Account: NAISS2026-4-280  |  GPU: A40  |  Env: defectgen_env
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

# --data_root        $DATA/cwp_pipeline_ready/genai/train \

#    --unet_lr          2e-4 \
#    --text_encoder_lr  4e-5 \

python -u train.py \
    --data_root        /mimer/NOBACKUP/groups/cast_fm/axel/_ARCHIVE/Data/MVTec_new/hazelnut/train \
    --defect_type      crack2 \
    --object_name      "hazelnut" \
    --output_dir       /mimer/NOBACKUP/groups/cast_fm/axel/Models/DefectGen1_output/output_attn_fix/hazelnut_default_1 \
    --pretrained_model_name sd2-community/stable-diffusion-2-inpainting \
    --train_steps      2000 \
    --train_fraction   0.9 \
    --save_steps       250 \
    --batch_size       4 \
    --unet_lr          2e-4 \
    --text_encoder_lr  4e-5 \
    --lora_rank        8 \
    --lora_alpha       16 \
    --lora_dropout     0.1 \
    --lambda_def       1.0 \
    --lambda_obj       0.1 \
    --lambda_attn      0.1 \
    --alpha            0.3 \
    --warmup_steps     100 \
    --mixed_precision  bf16 \
    --seed             42


