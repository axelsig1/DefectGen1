"""
DefectFill Loss Functions
=========================

Three complementary loss terms as described in the paper:

    L_ours = λ_def · L_def + λ_obj · L_obj + λ_attn · L_attn

References
----------
Equations (5), (7), (8) and (9) from the DefectFill paper.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention map extraction via processor monkey-patch
# ---------------------------------------------------------------------------

class AttnProbeProcessor:
    """
    Custom attention processor that captures cross-attention weight maps and
    produces the same hidden-state output as the default diffusers processor.

    Why manual softmax instead of ``attn.get_attention_scores()``
    -------------------------------------------------------------
    ``get_attention_scores`` is an internal diffusers method that internally
    branches on whether xformers / ``scaled_dot_product_attention`` (SDPA) is
    available.  When those memory-efficient backends are active they fuse the
    softmax into a CUDA kernel and *never materialise the full attention matrix*,
    so ``get_attention_scores`` either falls back to a slower path or raises an
    error depending on the diffusers version.

    The fix is to bypass that method entirely and compute standard scaled
    dot-product attention by hand:

        scores = Q · Kᵀ / √d_head
        probs  = softmax(scores + mask, dim=-1)
        out    = probs · V

    This always materialises the (batch*heads, spatial, text_len) probability
    matrix we need for the attention loss, regardless of which backend is
    installed, and produces bit-identical results to the non-fused path.
    """

    def __init__(self, store: list):
        self.store = store

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states

        # Optional group norm (present in some SD variants)
        if hasattr(attn, "group_norm") and attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states

        query = attn.to_q(hidden_states)
        key   = attn.to_k(kv_input)
        value = attn.to_v(kv_input)

        # Reshape: (B, seq, d_model) → (B*heads, seq, d_head)
        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # --- Manual scaled-dot-product softmax attention ---
        # This always materialises the full (bh, spatial, text) matrix,
        # regardless of xformers / SDPA availability.
        scale = query.shape[-1] ** -0.5
        # (bh, spatial, text) — use float32 for numerical stability
        sim = torch.bmm(query.float() * scale, key.float().transpose(-1, -2))

        if attention_mask is not None:
            # attention_mask shape can be (B, 1, seq_q, seq_k) or (B*heads, seq_q, seq_k)
            if attention_mask.ndim == 4:
                bh = sim.shape[0]
                attention_mask = attention_mask.reshape(bh, *attention_mask.shape[2:])
            sim = sim + attention_mask.float()

        attn_probs = torch.softmax(sim, dim=-1).to(query.dtype)  # (bh, spatial, text)

        # Store cross-attention maps only (not self-attention)
        if is_cross:
            self.store.append(attn_probs.detach())

        # Weighted sum over value vectors
        hidden_states = torch.bmm(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Output projection + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Residual connection
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# Defect Loss  (Eq. 5)
# ---------------------------------------------------------------------------

def defect_loss(
    noise_pred: torch.Tensor,
    noise_target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    L_def = E[ || M ⊙ (ε - ε_θ(x_t^def, t, c^def)) ||² ]

    Parameters
    ----------
    noise_pred   : (B, C, H, W) — model noise prediction
    noise_target : (B, C, H, W) — sampled Gaussian noise ε
    mask         : (B, 1, H, W) — binary defect mask M  ∈ {0, 1}
    """
    # Upsample mask to match noise resolution if needed
    if mask.shape[-2:] != noise_pred.shape[-2:]:
        mask = F.interpolate(mask, size=noise_pred.shape[-2:], mode="nearest")

    residual = noise_target - noise_pred          # (B, C, H, W)
    masked = mask * residual                       # apply defect mask
    return (masked ** 2).mean()


# ---------------------------------------------------------------------------
# Object Loss  (Eq. 7)
# ---------------------------------------------------------------------------

def object_loss(
    noise_pred: torch.Tensor,
    noise_target: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.3,
) -> torch.Tensor:
    """
    L_obj = E[ || M' ⊙ (ε - ε_θ(x_t^obj, t, c^obj)) ||² ]
    M' = M + α · (1 − M)

    Parameters
    ----------
    noise_pred   : (B, C, H, W)
    noise_target : (B, C, H, W)
    mask         : (B, 1, H, W) — defect mask M (NOT M_rand)
    alpha        : weight for background pixels (default 0.3)
    """
    if mask.shape[-2:] != noise_pred.shape[-2:]:
        mask = F.interpolate(mask, size=noise_pred.shape[-2:], mode="nearest")

    # Adjusted mask: M' = M + α*(1-M)
    mask_prime = mask + alpha * (1.0 - mask)      # (B, 1, H, W)

    residual = noise_target - noise_pred
    masked = mask_prime * residual
    return (masked ** 2).mean()


# ---------------------------------------------------------------------------
# Attention Loss  (Eq. 8)
# ---------------------------------------------------------------------------

def attention_loss(
    attn_map_vstar: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    L_attn = E[ || A_t^[V*] − M ||² ]

    Parameters
    ----------
    attn_map_vstar : (1, 1, latent_H, latent_W)  — normalised [V*] attention map
    mask           : (B, 1, H, W)                 — defect mask at image resolution
    """
    if mask.shape[-2:] != attn_map_vstar.shape[-2:]:
        mask = F.interpolate(
            mask.float(),
            size=attn_map_vstar.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    # Expand attn to batch if necessary
    if attn_map_vstar.shape[0] != mask.shape[0]:
        attn_map_vstar = attn_map_vstar.expand(mask.shape[0], -1, -1, -1)

    return ((attn_map_vstar - mask) ** 2).mean()


# ---------------------------------------------------------------------------
# Combined DefectFill Loss  (Eq. 9)
# ---------------------------------------------------------------------------

def defectfill_loss(
    noise_pred_def: torch.Tensor,
    noise_pred_obj: torch.Tensor,
    noise_target: torch.Tensor,
    mask: torch.Tensor,
    attn_map_vstar: Optional[torch.Tensor],
    alpha: float = 0.3,
    lambda_def: float = 0.5,
    lambda_obj: float = 0.2,
    lambda_attn: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """
    Compute all three DefectFill loss terms and their weighted combination.

    Parameters
    ----------
    noise_pred_def  : UNet noise prediction using defect branch (P_def)
    noise_pred_obj  : UNet noise prediction using object branch (P_obj)
    noise_target    : ground-truth noise ε
    mask            : defect mask M  (B, 1, H, W)
    attn_map_vstar  : [V*] cross-attention map from decoder layers  (or None)
    alpha           : background weight in M'
    lambda_*        : loss term weights

    Returns
    -------
    dict with keys: 'loss', 'l_def', 'l_obj', 'l_attn'
    """
    l_def = defect_loss(noise_pred_def, noise_target, mask)
    l_obj = object_loss(noise_pred_obj, noise_target, mask, alpha)

    if attn_map_vstar is not None:
        l_attn = attention_loss(attn_map_vstar, mask)
    else:
        l_attn = torch.tensor(0.0, device=noise_pred_def.device)

    total = lambda_def * l_def + lambda_obj * l_obj + lambda_attn * l_attn

    return {
        "loss": total,
        "l_def": l_def.detach(),
        "l_obj": l_obj.detach(),
        "l_attn": l_attn.detach() if isinstance(l_attn, torch.Tensor) else l_attn,
    }