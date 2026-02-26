"""
DefectFill Generation Script
=============================

Loads the fine-tuned LoRA weights and generates realistic defect images by
inpainting defect-free images with the learned defect concept.

For each (good image, mask) pair, *num_samples_lfs* candidate images are
generated and the one with the highest LPIPS score inside the mask is
selected (Low-Fidelity Selection).

Usage
-----
python generate.py \
    --pretrained_model_name sd2-community/stable-diffusion-2-inpainting \
    --lora_weights_path output/hazelnut_hole/unet_lora_final \
    --te_lora_weights_path output/hazelnut_hole/text_encoder_lora_final \
    --good_images_dir data/hazelnut/good \
    --masks_dir data/hazelnut/masks_for_generation \
    --output_dir generated/hazelnut_hole \
    --object_name hazelnut \
    --num_samples_lfs 8
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import lpips
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from config import GenerationConfig
from dataset import GoodImagesDataset, _list_images, _load_mask
from utils import (
    set_seed,
    make_object_prompt,
    tensor_to_pil,
    pil_to_tensor,
    low_fidelity_selection,
    save_image,
)

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(description="Generate defect images with DefectFill")
    parser.add_argument("--pretrained_model_name", type=str,
                        default="sd2-community/stable-diffusion-2-inpainting")
    parser.add_argument("--lora_weights_path", type=str, required=True,
                        help="Path to fine-tuned UNet LoRA weights directory.")
    parser.add_argument("--te_lora_weights_path", type=str, required=True,
                        help="Path to fine-tuned text encoder LoRA weights directory.")
    parser.add_argument("--good_images_dir", type=str, required=True,
                        help="Object root (e.g. data/concrete) or direct path "
                             "to good images. test/good/ is auto-resolved.")
    parser.add_argument("--masks_dir", type=str, default="",
                        help="Directory of binary masks. If empty, a random mask is generated.")
    parser.add_argument("--output_dir", type=str, default="generated")
    parser.add_argument("--object_name", type=str, required=True)
    parser.add_argument("--placeholder_token", type=str, default="sks")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_samples_lfs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])

    args = parser.parse_args()
    cfg = GenerationConfig()
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    # Extra args not in config
    cfg._lora_weights_path = args.lora_weights_path
    cfg._te_lora_weights_path = args.te_lora_weights_path
    return cfg


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(
    pretrained_model: str,
    unet_lora_path: str,
    te_lora_path: str,
    weight_dtype,
    device,
):
    """
    Load the SD2-inpainting pipeline and inject LoRA weights.
    """
    from diffusers import StableDiffusionInpaintPipeline
    from peft import PeftModel

    logger.info(f"Loading pipeline from: {pretrained_model}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )

    logger.info("Loading UNet LoRA weights …")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_lora_path)
    pipe.unet.merge_adapter()  # Merge LoRA into base weights for faster inference

    logger.info("Loading text encoder LoRA weights …")
    pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, te_lora_path)
    pipe.text_encoder.merge_adapter()

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    return pipe


# ---------------------------------------------------------------------------
# Blended latent diffusion inpainting
# ---------------------------------------------------------------------------

def generate_with_blended_latents(
    pipe,
    good_image: torch.Tensor,       # (3, H, W) in [-1, 1]
    mask: torch.Tensor,             # (1, H, W) in {0, 1}
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    generator: torch.Generator,
    device,
    weight_dtype,
) -> torch.Tensor:
    """
    Run inpainting with "blended latent diffusion" style background replacement:
    at each denoising step, the unmasked region is replaced with the noisy
    latent of the original image, preserving background integrity.

    Returns (3, H, W) tensor in [-1, 1].
    """
    H, W = good_image.shape[-2], good_image.shape[-1]
    img_pil = tensor_to_pil(good_image.cpu())
    mask_np = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np)

    result = pipe(
        prompt=prompt,
        image=img_pil,
        mask_image=mask_pil,
        height=H,
        width=W,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pt",
    ).images  # (1, 3, H, W) in [0, 1]

    # Convert [0,1] → [-1,1]
    return result[0] * 2.0 - 1.0    # (3, H, W)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(cfg: GenerationConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = (
        torch.float16 if cfg.mixed_precision == "fp16"
        else torch.bfloat16 if cfg.mixed_precision == "bf16"
        else torch.float32
    )

    # Build pipeline
    pipe = build_pipeline(
        cfg.pretrained_model_name,
        cfg._lora_weights_path,
        cfg._te_lora_weights_path,
        weight_dtype,
        device,
    )

    # LPIPS for Low-Fidelity Selection
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    # Object prompt
    prompt = make_object_prompt(cfg.object_name, cfg.placeholder_token)
    logger.info(f"Generation prompt: '{prompt}'")

    # Good images
    good_paths = _list_images(cfg.good_images_dir)
    logger.info(f"Found {len(good_paths)} good images in '{cfg.good_images_dir}'")

    # Masks
    if cfg.masks_dir:
        mask_paths = _list_images(cfg.masks_dir)
        logger.info(f"Found {len(mask_paths)} masks in '{cfg.masks_dir}'")
    else:
        mask_paths = []
        logger.info("No masks directory provided; random masks will be generated.")

    # Pair good images with masks (cycle through masks if fewer than images)
    pairs = []
    for i, gp in enumerate(good_paths):
        if mask_paths:
            mp = mask_paths[i % len(mask_paths)]
        else:
            mp = None
        pairs.append((gp, mp))

    # ------------------------------------------------------------------ #
    # Generation loop
    # ------------------------------------------------------------------ #
    global_idx = 0
    for img_path, mask_path in tqdm(pairs, desc="Generating"):
        img_pil = Image.open(img_path).convert("RGB").resize(
            (cfg.image_size, cfg.image_size), Image.LANCZOS
        )
        good_tensor = pil_to_tensor(img_pil).unsqueeze(0)   # (1,3,H,W)

        if mask_path is not None:
            mask_pil = _load_mask(mask_path, cfg.image_size)
        else:
            # Generate a simple random box mask centred in the image
            from dataset import generate_random_box_mask
            mask_t = generate_random_box_mask(cfg.image_size, cfg.image_size, num_boxes=1,
                                               min_frac=0.1, max_frac=0.3)
            mask_pil = Image.fromarray((mask_t.squeeze(0).numpy() * 255).astype(np.uint8))

        mask_arr = np.array(mask_pil)
        mask_tensor = torch.from_numpy(mask_arr).float().unsqueeze(0).unsqueeze(0) / 255.0  # (1,1,H,W)

        # Generate cfg.num_samples_lfs candidates
        candidates = []
        for s in range(cfg.num_samples_lfs):
            gen = torch.Generator(device=device).manual_seed(cfg.seed + global_idx * 100 + s)
            gen_img = generate_with_blended_latents(
                pipe=pipe,
                good_image=good_tensor.squeeze(0),
                mask=mask_tensor.squeeze(0),
                prompt=prompt,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                generator=gen,
                device=device,
                weight_dtype=weight_dtype,
            )
            candidates.append(gen_img.unsqueeze(0))   # (1,3,H,W)

        # Low-Fidelity Selection
        best_img, best_idx, best_score = low_fidelity_selection(
            lpips_fn=lpips_fn,
            generated_images=candidates,
            original_image=good_tensor.to(device),
            mask=mask_tensor.to(device),
        )

        # Save best image and its mask
        stem = Path(img_path).stem
        out_img_path = os.path.join(cfg.output_dir, f"{stem}_defect_{global_idx:04d}.png")
        out_mask_path = os.path.join(cfg.output_dir, f"{stem}_mask_{global_idx:04d}.png")

        save_image(best_img.squeeze(0).cpu(), out_img_path)
        mask_pil.save(out_mask_path)

        logger.debug(
            f"[{global_idx:04d}] {img_path.name}: "
            f"selected sample {best_idx} (LPIPS={best_score:.4f})"
        )
        global_idx += 1

    logger.info(f"Generated {global_idx} defect images → '{cfg.output_dir}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = parse_args()
    generate(cfg)
