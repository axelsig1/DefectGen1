#!/usr/bin/env bash
# =============================================================================
# DefectFill — Grid Search: checkpoint × CFG scale
# =============================================================================
# Tests 3 saved checkpoints against 3 CFG values → 9 output folders.
# Each folder contains exactly N images (one per test mask).
#
# Usage (from your interactive session):
#   bash grid_search.sh
# =============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="/mimer/NOBACKUP/groups/cast_fm/axel"
MODEL_DIR="$ROOT/Models/DefectGen1"
DATA="$ROOT/Data"
ENV="$ROOT/envs/defectgen_env"

source "$ENV/bin/activate"
cd "$MODEL_DIR"

# ---------- edit these to match your run ----------
OBJECT="hazelnut"
DEFECT="crack"
TRAIN_RUN="/mimer/NOBACKUP/groups/cast_fm/axel/Models/DefectGen1_output/output/cwp_uv_model_2"          # folder from train.py --output_dir
DATA_ROOT="$DATA/MVTec_new/$OBJECT"
GOOD_DIR="$DATA/MVTec_new/$OBJECT/test/good"
MODEL="sd2-community/stable-diffusion-2-inpainting"
SEED=42
LFS_SAMPLES=4                                  # 4 samples per mask (was 8)
STEPS=50
# ---------------------------------------------------

CHECKPOINTS=("750" "1000" "1250" "1500" "1750" "final")
CFGS=(2.5 5.0)

echo "Grid search: ${#CHECKPOINTS[@]} checkpoints × ${#CFGS[@]} CFG values"
echo "LFS samples per mask : $LFS_SAMPLES"
echo "Output root          : $MODEL_DIR/output/grid_search/"
echo ""

for CKPT in "${CHECKPOINTS[@]}"; do
    UNET_PATH="$TRAIN_RUN/unet_lora_${CKPT}"
    TE_PATH="$TRAIN_RUN/text_encoder_lora_${CKPT}"

    if [ ! -d "$UNET_PATH" ]; then
        echo "SKIP: checkpoint '$CKPT' not found at $UNET_PATH"
        continue
    fi

    for CFG in "${CFGS[@]}"; do
        OUT="/mimer/NOBACKUP/groups/cast_fm/axel/Models/DefectGen1_output/generated/grid_search/${DEFECT}_${CKPT}_cfg${CFG}"
        echo "================================================="
        echo "Checkpoint : $CKPT   CFG : $CFG"
        echo "Output     : $OUT"
        echo "================================================="

        python -u generate.py \
            --pretrained_model_name  "$MODEL" \
            --lora_weights_path      "$UNET_PATH" \
            --te_lora_weights_path   "$TE_PATH" \
            --good_images_dir        "$GOOD_DIR" \
            --data_root              "$DATA_ROOT" \
            --defect_type            "$DEFECT" \
            --output_dir             "$OUT" \
            --object_name            "$OBJECT" \
            --num_inference_steps    $STEPS \
            --guidance_scale         $CFG \
            --num_samples_lfs        $LFS_SAMPLES \
            --mixed_precision        bf16 \
            --seed                   $SEED \
            --mask_dilation_size     0 \
            --mask_blur_radius       2

        echo "Done: $OUT"
        echo ""
    done
done

echo "================================================="
echo "Grid search complete."
echo "Results in: $MODEL_DIR/output/grid_search/"
echo "================================================="