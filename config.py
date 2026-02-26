"""
DefectFill Configuration
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    # ---------- Data ----------
    data_root: str = "data"
    """Root directory containing 'defective', 'defective_masks', and 'good' sub-folders."""

    object_name: str = "hazelnut"
    """Human-readable name of the object, used in the object prompt P_obj."""

    placeholder_token: str = "sks"
    """The [V*] special token that represents the defect concept."""

    image_size: int = 512
    """Target image resolution (height == width)."""

    # ---------- LoRA ----------
    lora_rank: int = 8
    lora_alpha: int = 16          # scaling factor
    lora_dropout: float = 0.1

    # ---------- Optimisation ----------
    train_steps: int = 2000
    warmup_steps: int = 100
    batch_size: int = 4

    unet_lr: float = 2e-4
    text_encoder_lr: float = 4e-5

    # ---------- DefectFill loss weights ----------
    lambda_def: float = 0.5
    lambda_obj: float = 0.2
    lambda_attn: float = 0.05

    # ---------- Object-loss adjusted mask ----------
    alpha: float = 0.3
    """Weight applied to background pixels in M' for the object loss."""

    # ---------- Augmentation ----------
    resize_min: float = 1.0
    resize_max: float = 1.125
    """Random resize factor range; image+mask are jointly scaled then centre-cropped."""

    # ---------- Random mask (M_rand) ----------
    num_rand_boxes: int = 30
    rand_box_min_frac: float = 0.03
    rand_box_max_frac: float = 0.25
    """Box side length as fraction of the image size."""

    # ---------- Inference / generation ----------
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_samples_lfs: int = 8
    """Number of samples to generate per (image, mask) pair for Low-Fidelity Selection."""

    # ---------- Misc ----------
    output_dir: str = "output"
    seed: int = 42
    mixed_precision: str = "fp16"   # "no" | "fp16" | "bf16"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    save_steps: int = 500
    log_steps: int = 50

    pretrained_model_name: str = "stabilityai/stable-diffusion-2-inpainting"


@dataclass
class GenerationConfig:
    # Model / LoRA checkpoint
    pretrained_model_name: str = "stabilityai/stable-diffusion-2-inpainting"
    lora_weights_path: str = "output/lora_weights"

    object_name: str = "hazelnut"
    placeholder_token: str = "sks"

    # Generation inputs
    good_images_dir: str = "data/good"
    masks_dir: str = ""
    """Directory of binary PNG masks (0/255). If empty, masks are drawn interactively."""

    output_dir: str = "generated"
    image_size: int = 512

    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_samples_lfs: int = 8
    """Samples per (image, mask) pair; best by LPIPS is kept."""

    seed: int = 42
    mixed_precision: str = "fp16"
