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
# Low-Fidelity Selection (LFS)
# ---------------------------------------------------------------------------

def _mask_bounding_box(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    Return the tight axis-aligned bounding box of the non-zero region in a
    (1, 1, H, W) or (1, H, W) binary mask tensor.

    Returns (y_min, y_max, x_min, x_max) with inclusive bounds.
    Falls back to the full image if the mask is entirely zero.
    """
    m = mask.squeeze()  # → (H, W)
    rows = torch.any(m > 0, dim=1)
    cols = torch.any(m > 0, dim=0)

    if not rows.any():
        H, W = m.shape
        return 0, H - 1, 0, W - 1

    y_min = int(rows.nonzero(as_tuple=False)[0].item())
    y_max = int(rows.nonzero(as_tuple=False)[-1].item())
    x_min = int(cols.nonzero(as_tuple=False)[0].item())
    x_max = int(cols.nonzero(as_tuple=False)[-1].item())
    return y_min, y_max, x_min, x_max


def compute_lpips_in_mask(
    lpips_fn,
    generated: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Compute LPIPS between *generated* and *original* restricted to the masked
    region, using a **bounding-box crop** rather than zeroing the background.

    Why bounding-box crop instead of masking-to-zero
    -------------------------------------------------
    LPIPS uses deep convolutional networks (AlexNet / VGG).  Multiplying the
    image by a binary mask creates a hard black border at the mask boundary.
    Every convolution kernel that straddles that border picks up this
    artificial edge, injecting a spurious signal that inflates or deflates the
    LPIPS score depending on boundary shape — completely unrelated to how
    well the defect was generated.

    Cropping to the tight bounding box of the mask region removes the
    artificial edge entirely: the convolutions see only real image content.
    This also matches Figure 3 of the paper, which visually shows the defect
    patch in isolation.

    All tensors are (1, 3, H, W) in [-1, 1]; mask is (1, 1, H, W) in {0, 1}.
    """
    y_min, y_max, x_min, x_max = _mask_bounding_box(mask)

    # Physical crop — no artificial edges, only real pixel content
    gen_crop  = generated[:, :, y_min : y_max + 1, x_min : x_max + 1]
    orig_crop = original[:, :,  y_min : y_max + 1, x_min : x_max + 1]

    # LPIPS requires at least a minimal spatial size; guard against tiny crops
    if gen_crop.shape[-1] < 16 or gen_crop.shape[-2] < 16:
        gen_crop  = F.interpolate(gen_crop,  size=(16, 16), mode="bilinear", align_corners=False)
        orig_crop = F.interpolate(orig_crop, size=(16, 16), mode="bilinear", align_corners=False)

    # lpips_fn lives on GPU; crops may be on CPU (e.g. from pil_to_tensor).
    # Move both to wherever lpips_fn is before calling forward.
    lpips_device = next(lpips_fn.parameters()).device
    with torch.no_grad():
        score = lpips_fn(
            gen_crop.to(lpips_device),
            orig_crop.to(lpips_device),
        ).item()
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
    pronounced defect expression).

    Parameters
    ----------
    lpips_fn         : lpips.LPIPS instance
    generated_images : list of (1, 3, H, W) tensors in [-1, 1]
    original_image   : (1, 3, H, W) tensor in [-1, 1]
    mask             : (1, 1, H, W) binary mask

    Returns
    -------
    best_image : the selected tensor
    best_idx   : its index in the input list
    best_score : its LPIPS score
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