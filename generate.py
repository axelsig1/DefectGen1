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
from PIL import ImageFilter

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
    parser.add_argument("--data_root", type=str, default="",
                        help="Object root used during training (e.g. data/hazelnut). "
                             "When provided, the test-split masks (the 2/3 not used for "
                             "training) are loaded automatically using the same split logic "
                             "as train.py. Takes priority over --masks_dir.")
    parser.add_argument("--masks_dir", type=str, default="",
                        help="Explicit directory of binary masks. Ignored when --data_root "
                             "is set. If both are empty, random masks are generated.")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Must match the seed used during training (default 42).")
    parser.add_argument("--train_fraction", type=float, default=0.3333,
                        help="Must match the fraction used during training (default 1/3).")
    parser.add_argument("--defect_type", type=str, default=None,
                        help="Defect subfolder, e.g. crack. Must match train.py.")
    parser.add_argument("--output_dir", type=str, default="generated")
    parser.add_argument("--object_name", type=str, required=True)
    parser.add_argument("--placeholder_token", type=str, default="sks")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_samples_lfs", type=int, default=8)
    parser.add_argument("--mask_dilation_size", type=int, default=0, help="Thickens the mask to prevent losing thin defects.")
    parser.add_argument("--mask_blur_radius", type=int, default=0, help="Blurs the mask edges for seamless blending.")
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
    # These three are now proper fields on GenerationConfig so the
    # hasattr loop above already copies them — but set explicitly to
    # be safe and for clarity.
    cfg.data_root = args.data_root
    cfg.split_seed = args.split_seed
    cfg.train_fraction = args.train_fraction
    cfg.defect_type = args.defect_type
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

    dtype strategy
    --------------
    The LoRA adapters were trained with the UNet and text encoder in fp32.
    Loading the pipeline in fp16 and then calling merge_adapter() would merge
    fp32 LoRA deltas into fp16 base weights — the merge rounds to fp16,
    discarding the fine numerical detail that LoRA learned, and the fp16
    activations at inference are in a different numerical regime than the fp32
    activations the adapters were tuned against.  The result is corrupted,
    "deep-fried" outputs.

    Fix: load the pipeline in fp32 (the A40 has 48 GB, fp32 fits easily), apply
    the PEFT LoRA on top without merging, and use torch.autocast only for the
    actual denoising loop to keep inference fast.  Never call merge_adapter().
    """
    from diffusers import StableDiffusionInpaintPipeline
    from peft import PeftModel

    logger.info(f"Loading pipeline from: {pretrained_model}")
    # Load base pipeline in fp32 — dtype must match what LoRA was trained with.
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=torch.float32,   # fp32, same as training
        safety_checker=None,
    )

    logger.info("Loading UNet LoRA weights …")
    # Apply LoRA on top of the fp32 base — do NOT call merge_adapter().
    pipe.unet = PeftModel.from_pretrained(
        pipe.unet, unet_lora_path, is_trainable=False
    )

    logger.info("Loading text encoder LoRA weights …")
    pipe.text_encoder = PeftModel.from_pretrained(
        pipe.text_encoder, te_lora_path, is_trainable=False
    )

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
    Run SD2 inpainting and return a (3, H, W) tensor in [-1, 1].

    Normalisation contract
    ----------------------
    Input  good_image : (3, H, W) float32 tensor in [-1, 1]
    Input  mask       : (1, H, W) float32 tensor in {0, 1}  (1 = inpaint here)

    The pipeline receives a PIL image (tensor_to_pil handles [-1,1] → [0,255])
    and a PIL mask.  We use output_type="pil" deliberately:

      output_type="pt"  is unreliable across diffusers versions — some return
      the raw VAE decode in ~[-1,1], others normalise to [0,1].  Applying
      * 2 - 1 to a [-1,1] tensor gives [-3, 1], which produces the blown-out,
      over-saturated "deep-fried" artefacts.

      output_type="pil" always gives a uint8 PIL image in [0, 255] regardless
      of diffusers version.  We convert back to [-1,1] via pil_to_tensor.

    The pipe is loaded in fp32 (LoRA was trained in fp32).  We wrap the call
    in autocast so the denoising steps run in fp16 for speed while the model
    weights stay in fp32.
    """
    H, W = good_image.shape[-2], good_image.shape[-1]

    # [-1,1] tensor → PIL [0,255]  (pipeline always takes PIL)
    img_pil  = tensor_to_pil(good_image.cpu())
    mask_np  = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np)

    # Run pipeline in fp32, no outer autocast.
    #
    # Why no autocast:
    #   The pipeline was loaded with torch_dtype=float32.  Wrapping it in an
    #   autocast(fp16) context makes diffusers' internal dtype guards see fp16
    #   activations where they expect fp32, which corrupts the VAE decode and
    #   produces the wrong-colour "deep-fried" artefacts.
    #
    # Why output_type="pil":
    #   output_type="pt" is unreliable across diffusers versions (some return
    #   the raw VAE output in ~[-1,1]; others return normalised [0,1]).
    #   PIL always gives a clean uint8 [0,255] image.
    result_pil = pipe(
        prompt=prompt,
        image=img_pil,
        mask_image=mask_pil,
        height=H,
        width=W,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pil",
    ).images[0]   # PIL Image [0, 255]

    # PIL [0,255] → float tensor [-1,1]
    result = pil_to_tensor(result_pil).to(good_image.device)  # (3, H, W)

    # -------------------------------------------------------------------
    # Background paste-back (critical for inpainting quality)
    # -------------------------------------------------------------------
    # The pipeline decodes the full latent through the VAE at the end.
    # VAE encode→decode is lossy: background pixels drift slightly, and
    # strong LoRA defect signals bleed into surrounding areas through the
    # convolutional layers.  Explicitly restoring the original pixels in
    # the non-masked region gives pixel-perfect background preservation,
    # which is also what the paper means by "replacing background latents
    # during inference".
    #
    #   mask  : (1, H, W) in {0, 1} — 1 = defect region, 0 = background
    #   result_final = result  × mask  +  original × (1 − mask)
    mask_3ch = mask.to(result.device).expand_as(result)   # (3, H, W)
    result = result * mask_3ch + good_image.to(result.device) * (1.0 - mask_3ch)

    return result    # (3, H, W) in [-1, 1]


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

    # ------------------------------------------------------------------ #
    # Resolve good images
    # ------------------------------------------------------------------ #
    from dataset import GoodImagesDataset, _resolve_good_dir, _list_images as _li
    good_dir = str(_resolve_good_dir(cfg.good_images_dir))
    good_paths = _li(good_dir)
    logger.info(f"Found {len(good_paths)} good images in '{good_dir}'")

    # ------------------------------------------------------------------ #
    # Resolve masks — prefer --data_root (uses exact test split) over
    # --masks_dir (manual) over random generation
    # ------------------------------------------------------------------ #
    data_root = getattr(cfg, "data_root", "")
    split_seed = getattr(cfg, "split_seed", 42)
    train_fraction = getattr(cfg, "train_fraction", 1.0 / 3.0)

    if data_root:
        # Reuse DefectFillDataset(split="test") directly — this is the exact
        # same code path that found 70 pairs during training, so it is
        # guaranteed to produce the right test masks without any duplication.
        from dataset import DefectFillDataset
        test_ds = DefectFillDataset(
            data_root=data_root,
            split="test",
            defect_type=getattr(cfg, "defect_type", None),
            train_fraction=train_fraction,
            split_seed=split_seed,
            image_size=cfg.image_size,
            augment=False,   # no augmentation needed; we only want the mask paths
        )
        mask_paths = [mp for _, mp in test_ds.pairs]
        logger.info(
            f"Using --data_root split: {len(mask_paths)} test masks "
            f"(train_fraction={train_fraction:.3f}, seed={split_seed})"
        )
    elif cfg.masks_dir:
        mask_paths = [Path(p) for p in _li(cfg.masks_dir)]
        logger.info(f"Found {len(mask_paths)} masks in '{cfg.masks_dir}'")
    else:
        mask_paths = []
        logger.info("No masks provided; random masks will be generated.")

    # Pair masks 1-to-1 with good images.
    # We have N masks and potentially many more good images.  Rather than
    # cycling masks over all good images (which would generate len(good_paths)
    # samples and apply each mask ~30 times), we randomly select exactly N
    # good images — one per mask — so the total output is N images.
    import random as _random
    import itertools

    _random.seed(cfg.seed)
    if mask_paths:
        # ORIGINAL CODE
        #n = len(mask_paths)
        #selected_good = _random.sample(good_paths, min(n, len(good_paths)))
        #pairs = list(zip(selected_good, mask_paths))
        #logger.info(f"Paired {len(pairs)} masks with {len(pairs)} randomly selected good images.")
        
        # VERSION 2:
        # Create a Cartesian product: Every good image paired with every mask
        #pairs = list(itertools.product(good_paths, mask_paths))
        #logger.info(f"Created {len(pairs)} combinations from {len(good_paths)} good images and {len(mask_paths)} masks.")

        # VERSION 3:
        # --- 1. Filter masks down to unique physical cracks ---
        unique_mask_paths = []
        seen_cracks = set()

        for path in mask_paths:
            # e.g., "loc1_uvS1_he_crack_00"
            stem = path.stem
            parts = stem.split('_')

            # Ensure the filename has enough parts to be tokenized correctly
            if len(parts) >= 5:
                # Unique fingerprint: Surface + DefectType + DefectID (e.g., "loc1_crack_00")
                unique_id = f"{parts[0]}_{parts[-2]}_{parts[-1]}"
            else:
                # Fallback just in case a file has a weird name
                unique_id = stem

            if unique_id not in seen_cracks:
                seen_cracks.add(unique_id)
                unique_mask_paths.append(path)

        logger.info(f"Filtered {len(mask_paths)} lighting variations down to {len(unique_mask_paths)} unique physical cracks.")

        # --- 2. Create the Cartesian Product ---
        # Now it pairs every good image ONLY with the 18 unique masks
        pairs = list(itertools.product(good_paths, unique_mask_paths))
        logger.info(f"Created {len(pairs)} combinations from {len(good_paths)} good images and {len(unique_mask_paths)} unique masks.")
    else:
        # No masks: generate one sample per good image with a random mask each
        pairs = [(gp, None) for gp in good_paths]
        logger.info(f"No masks; will generate random masks for {len(pairs)} good images.")

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

        # Dilation (Thickening): Expands the white areas so thin cracks aren't lost
        if cfg.mask_dilation_size > 0:
            mask_pil = mask_pil.filter(ImageFilter.MaxFilter(size=cfg.mask_dilation_size))

        # Gaussian Blur: Softens the edges into gradients for seamless blending
        if cfg.mask_blur_radius > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=cfg.mask_blur_radius))

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

        # Extract the stems for both background and mask
        bg_stem = Path(img_path).stem
        mask_stem = Path(mask_path).stem if mask_path is not None else "random"

        # Create the comprehensive filename
        out_img_name = f"{bg_stem}_WITH_{mask_stem}_defect_{global_idx:04d}.png"
        out_img_path = os.path.join(cfg.output_dir, out_img_name)

        # Save best image and its mask
        #stem = Path(img_path).stem
        #out_img_path = os.path.join(cfg.output_dir, f"{stem}_defect_{global_idx:04d}.png")
        out_mask_path = os.path.join(cfg.output_dir, f"{bg_stem}_WITH_{mask_stem}_mask_{global_idx:04d}.png")

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