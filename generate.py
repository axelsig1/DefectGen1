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

import matplotlib.pyplot as plt
import torch.nn.functional as F
from losses import AttnProbeProcessor

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
    parser.add_argument("--max_generations", type=int, default=0, help="Max images to generate (0 for all)")
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
    Initializes the Stable Diffusion inpainting pipeline and loads PEFT adapters.

    Args:
        pretrained_model (str): Path or HuggingFace ID for the base model.
        unet_lora_path (str): Path to the trained UNet LoRA weights.
        te_lora_path (str): Path to the trained Text Encoder LoRA weights.
        weight_dtype (torch.dtype): Execution precision (e.g., torch.float32).
        device (str): Target device ('cuda' or 'cpu').

    Returns:
        StableDiffusionInpaintPipeline: The fully loaded pipeline ready for inference.
    """
    from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
    from peft import PeftModel

    logger.info(f"Loading pipeline from: {pretrained_model}")

    # NEW better and faster DPM-solver instead of SD2 standard DDIM/PNDM
    dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        pretrained_model, subfolder="scheduler"
    )

    # Load base pipeline in fp32 — dtype must match what LoRA was trained with.
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        pretrained_model,
        scheduler=dpm_scheduler,    # Inject DPM
        torch_dtype=torch.float32,   # fp32, same as training
        safety_checker=None,
    )

    # Apply LoRA weights
    # Unet LoRA
    logger.info("Loading UNet LoRA weights …")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_lora_path, is_trainable=False)
    # Text Encoder LoRA
    logger.info("Loading text encoder LoRA weights …")
    pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, te_lora_path, is_trainable=False)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    return pipe


# ---------------------------------------------------------------------------
# Blended latent diffusion inpainting
# ---------------------------------------------------------------------------

def generate_with_blended_latents(
    pipe,
    good_image: torch.Tensor,
    mask: torch.Tensor,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    generator: torch.Generator,
    device,
    weight_dtype,
    placeholder_token: str = "sks",
) -> tuple[torch.Tensor, np.ndarray]:
    
    H, W = good_image.shape[-2], good_image.shape[-1]
    img_pil  = tensor_to_pil(good_image.cpu())
    mask_np  = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np)

    # 1. Setup Attention Probes
    attn_store = []

    def set_attn_processors(model, store):
        for name, module in model.named_modules():
            if "up_blocks" in name and hasattr(module, "set_processor"):
                module.set_processor(AttnProbeProcessor(store))

    def reset_attn_processors(model):
        from diffusers.models.attention_processor import AttnProcessor2_0
        for name, module in model.named_modules():
            if "up_blocks" in name and hasattr(module, "set_processor"):
                module.set_processor(AttnProcessor2_0())

    set_attn_processors(pipe.unet, attn_store)

    # 2. Run Pipeline
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
    ).images[0]

    reset_attn_processors(pipe.unet)

    # 3. Extract the Map
    token_ids = pipe.tokenizer.encode(prompt)
    sks_id = pipe.tokenizer.convert_tokens_to_ids(placeholder_token)
    try:
        vstar_idx = token_ids.index(sks_id)
    except ValueError:
        vstar_idx = -1

    heatmap = None
    if vstar_idx != -1 and len(attn_store) > 0:
        maps = []
        for attn_w in attn_store:
            bh, spatial, text_len = attn_w.shape
            if vstar_idx >= text_len: continue
            
            side = int(spatial ** 0.5)
            if side * side != spatial: continue
            
            # CFG duplicates the batch (unconditional + conditional)
            batch_size = 2 
            if bh % batch_size != 0: continue
                
            heads = bh // batch_size
            token_map = attn_w[:, :, vstar_idx].view(batch_size, heads, side, side)
            
            # Extract conditional pass (index 1) and average across heads
            cond_map = token_map[1:2].mean(dim=1, keepdim=True)
            cond_map = F.interpolate(
                cond_map.float(), size=(H, W), mode="bilinear", align_corners=False
            )
            maps.append(cond_map)
            
        if maps:
            # Average across decoder layers and steps, then normalize
            avg_map = torch.stack(maps, dim=0).mean(0)
            _min = avg_map.amin(dim=(-2, -1), keepdim=True)
            _max = avg_map.amax(dim=(-2, -1), keepdim=True)
            avg_map = (avg_map - _min) / (_max - _min + 1e-8)
            heatmap = avg_map.squeeze().cpu().numpy()

    attn_store.clear()

    # 4. Background paste-back
    result = pil_to_tensor(result_pil).to(good_image.device)
    mask_3ch = mask.to(result.device).expand_as(result)
    result = result * mask_3ch + good_image.to(result.device) * (1.0 - mask_3ch)

    return result, heatmap

# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(cfg: GenerationConfig):
    """
    Main execution pipeline: Initializes the model, iterates over the dataset, 
    generates defect candidates, and saves the best results.
    """
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

    # LPIPS for Low-Fidelity Selection (ONLY load if we are actually comparing samples)
    if cfg.num_samples_lfs > 1:
        lpips_fn = lpips.LPIPS(net="alex").to(device)
    else:
        lpips_fn = None

    # Object prompt
    prompt = make_object_prompt(cfg.object_name, cfg.placeholder_token)
    logger.info(f"Generation prompt: '{prompt}'")

    # Resolve good images
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
        # VERSION 1:
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

    if cfg.max_generations > 0:
        pairs = pairs[:cfg.max_generations]
        logger.info(f"Limiting generation to {cfg.max_generations} images.")

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

        # Force at least 1 sample to avoid crashing if you set it to 0
        num_samples = max(1, cfg.num_samples_lfs) 
        
        candidates = []
        candidate_heatmaps = [] 
        
        for s in range(num_samples):
            gen = torch.Generator(device=device).manual_seed(cfg.seed + global_idx * 100 + s)
            
            gen_img, heatmap = generate_with_blended_latents(
                pipe=pipe,
                good_image=good_tensor.squeeze(0),
                mask=mask_tensor.squeeze(0),
                prompt=prompt,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                generator=gen,
                device=device,
                weight_dtype=weight_dtype,
                placeholder_token=cfg.placeholder_token,
            )
            candidates.append(gen_img.unsqueeze(0))   # (1,3,H,W)
            candidate_heatmaps.append(heatmap)

        # -------------------------------------------------------------
        # NEW: Bypass LFS entirely if we only generated 1 sample
        # -------------------------------------------------------------
        if num_samples > 1:
            best_img, best_idx, best_score = low_fidelity_selection(
                lpips_fn=lpips_fn,
                generated_images=candidates,
                original_image=good_tensor.to(device),
                mask=mask_tensor.to(device),
            )
            best_heatmap = candidate_heatmaps[best_idx]
            logger.debug(f"[{global_idx:04d}] {img_path.name}: selected sample {best_idx} (LPIPS={best_score:.4f})")
        else:
            # We only have one image, so it wins by default!
            best_img = candidates[0]
            best_heatmap = candidate_heatmaps[0]
            logger.debug(f"[{global_idx:04d}] {img_path.name}: LFS bypassed (single sample)")

        # -------------------------------------------------------------
        # File Saving Logic
        # -------------------------------------------------------------
        bg_stem = Path(img_path).stem
        mask_stem = Path(mask_path).stem if mask_path is not None else "random"

        out_img_name = f"{bg_stem}_WITH_{mask_stem}_defect_{global_idx:04d}.png"
        out_img_path = os.path.join(cfg.output_dir, out_img_name)
        out_mask_path = os.path.join(cfg.output_dir, f"{bg_stem}_WITH_{mask_stem}_mask_{global_idx:04d}.png")
        
        # Save best image and its mask
        save_image(best_img.squeeze(0).cpu(), out_img_path)
        mask_pil.save(out_mask_path)
        
        # Plot and save the attention map overlay
        if best_heatmap is not None:
            # 1. Save the pure heatmap (no background, full opacity)
            heatmap_out_path = os.path.join(cfg.output_dir, f"{bg_stem}_WITH_{mask_stem}_heatmap_{global_idx:04d}.png")
            plt.imsave(heatmap_out_path, best_heatmap, cmap='jet')

            # 2. Save the overlay context (original image + semi-transparent heatmap)
            plt.figure(figsize=(6, 6))
            plt.imshow(img_pil) # Using original unmodified image as base 
            plt.imshow(best_heatmap, cmap='jet', alpha=0.55)
            plt.title(f"Attention Map for '{cfg.placeholder_token}'")
            plt.axis('off')
            
            overlay_out_path = os.path.join(cfg.output_dir, f"{bg_stem}_WITH_{mask_stem}_overlay_{global_idx:04d}.png")
            plt.savefig(overlay_out_path, bbox_inches='tight', dpi=150)
            plt.close()

        global_idx += 1

    logger.info(f"Generated {global_idx} defect images → '{cfg.output_dir}'")


if __name__ == "__main__":
    cfg = parse_args()
    generate(cfg)
