"""
DefectFill Utilities
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def make_defect_prompt(placeholder_token: str = "sks") -> str:
    """P_def: 'A photo of [V*]'"""
    return f"A photo of {placeholder_token}"


def make_object_prompt(object_name: str, placeholder_token: str = "sks") -> str:
    """P_obj: 'A [Object] with [V*]'"""
    return f"A {object_name} with {placeholder_token}"


# ---------------------------------------------------------------------------
# Token index lookup
# ---------------------------------------------------------------------------

def get_token_index(tokenizer, placeholder_token: str, prompt: str) -> int:
    """
    Return the position of *placeholder_token* within the tokenised *prompt*.
    Falls back to -2 (second-to-last before EOS) if not found.
    """
    tokens = tokenizer.encode(prompt)
    vocab = tokenizer.get_vocab()
    token_id = vocab.get(placeholder_token)
    if token_id is None:
        # Try encoding just the token
        enc = tokenizer.encode(placeholder_token, add_special_tokens=False)
        token_id = enc[0] if enc else None

    if token_id is not None and token_id in tokens:
        return tokens.index(token_id)

    # Heuristic: placeholder is usually the last content token before EOS
    return len(tokens) - 2


# ---------------------------------------------------------------------------
# LoRA application helpers
# ---------------------------------------------------------------------------

def add_lora_to_unet(unet, rank: int = 8, alpha: int = 16, dropout: float = 0.1):
    """
    Apply LoRA to the UNet's attention projection matrices (Q, K, V, out)
    in both self-attention and cross-attention layers.
    """
    from peft import LoraConfig, get_peft_model

    target_modules = []
    for name, module in unet.named_modules():
        if hasattr(module, "to_q") or hasattr(module, "to_k"):
            # This is an attention block; add sub-module names
            for sub in ["to_q", "to_k", "to_v", "to_out.0"]:
                full_name = f"{name}.{sub}" if name else sub
                target_modules.append(full_name)

    # Deduplicate
    target_modules = list(dict.fromkeys(target_modules))

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    return unet


def add_lora_to_text_encoder(
    text_encoder, rank: int = 8, alpha: int = 16, dropout: float = 0.1
):
    """
    Apply LoRA to the text encoder's projection matrices.
    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=dropout,
        bias="none",
    )
    text_encoder = get_peft_model(text_encoder, lora_config)
    return text_encoder


# ---------------------------------------------------------------------------
# Latent <-> image conversions
# ---------------------------------------------------------------------------

def encode_image(vae, image_tensor: torch.Tensor, weight_dtype) -> torch.Tensor:
    """Encode (B, 3, H, W) image tensor to latent using VAE encoder."""
    with torch.no_grad():
        latent = vae.encode(image_tensor.to(dtype=weight_dtype)).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
    return latent


def decode_latent(vae, latent: torch.Tensor) -> torch.Tensor:
    """Decode latent to (B, 3, H, W) image tensor in [-1, 1]."""
    with torch.no_grad():
        image = vae.decode(latent / vae.config.scaling_factor).sample
    return image


# ---------------------------------------------------------------------------
# Prepare inpainting input tensors
# ---------------------------------------------------------------------------

def prepare_inpaint_latents(
    vae,
    image: torch.Tensor,
    background: torch.Tensor,
    mask: torch.Tensor,
    noise_scheduler,
    timestep: torch.Tensor,
    noise: torch.Tensor,
    weight_dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build x_t^inpaint = concat(x_t, b, M_resized).

    Returns
    -------
    x_t         : noisy latent
    b_latent    : background latent
    mask_latent : mask downsampled to latent resolution
    """
    x0 = encode_image(vae, image, weight_dtype)
    b_latent = encode_image(vae, background, weight_dtype)

    # Add noise at the chosen timestep
    x_t = noise_scheduler.add_noise(x0, noise, timestep)

    # Resize mask to latent size
    latent_h = x0.shape[-2]
    latent_w = x0.shape[-1]
    mask_latent = F.interpolate(
        mask.float(), size=(latent_h, latent_w), mode="nearest"
    ).to(dtype=weight_dtype)

    return x_t, b_latent, mask_latent


# ---------------------------------------------------------------------------
# Low-Fidelity Selection (LFS)
# ---------------------------------------------------------------------------

def compute_lpips_in_mask(
    lpips_fn,
    generated: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Compute LPIPS between *generated* and *original* images, restricted to
    the masked region.

    Both tensors are (1, 3, H, W) in [-1, 1].
    mask is (1, 1, H, W) in {0, 1}.
    """
    # Crop to bounding box of mask for efficiency (optional)
    mask_3 = mask.expand_as(generated)
    # Zero out everything outside the mask before computing LPIPS
    gen_masked = generated * mask_3
    orig_masked = original * mask_3

    with torch.no_grad():
        score = lpips_fn(gen_masked, orig_masked).item()
    return score


def low_fidelity_selection(
    lpips_fn,
    generated_images: List[torch.Tensor],
    original_image: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, int, float]:
    """
    From a list of generated images, select the one with the *highest* LPIPS
    score within the masked region (= lowest reconstruction fidelity = most
    pronounced defect).

    Parameters
    ----------
    lpips_fn         : lpips.LPIPS instance
    generated_images : list of (1, 3, H, W) tensors in [-1, 1]
    original_image   : (1, 3, H, W) tensor in [-1, 1]
    mask             : (1, 1, H, W) binary mask

    Returns
    -------
    best_image : selected tensor
    best_idx   : index in the input list
    best_score : LPIPS score of the selected image
    """
    best_score = -1.0
    best_idx = 0
    for i, gen in enumerate(generated_images):
        score = compute_lpips_in_mask(lpips_fn, gen, original_image, mask)
        if score > best_score:
            best_score = score
            best_idx = i

    return generated_images[best_idx], best_idx, best_score


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert (3, H, W) tensor in [-1, 1] to PIL RGB image."""
    img = (tensor.float().cpu().clamp(-1, 1) + 1.0) / 2.0
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL RGB image to (3, H, W) float tensor in [-1, 1]."""
    import torchvision.transforms.functional as TF
    return TF.to_tensor(img) * 2.0 - 1.0


def save_image(tensor: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tensor_to_pil(tensor).save(path)


# ---------------------------------------------------------------------------
# Learning rate scheduler with linear warmup
# ---------------------------------------------------------------------------

def get_linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Linear warmup from 0 → 1 over warmup_steps, then constant at 1.
    """
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)
