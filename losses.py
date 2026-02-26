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
# Cross-attention map extraction utilities
# ---------------------------------------------------------------------------

class AttentionStore:
    """
    Hook manager that collects cross-attention maps from UNet *decoder* layers.

    Usage
    -----
    store = AttentionStore()
    store.register(unet)
    # run unet forward pass …
    attn_maps = store.get_and_clear()
    store.remove()
    """

    def __init__(self):
        self._hooks = []
        self._maps: list = []

    def _hook_fn(self, module, input, output):
        # Attention processors return tensors or tuples; we want the attn weights.
        # For diffusers' Attention module the weights are accessible via
        # `output` when `return_attention_probs` is used, but in standard
        # inference we tap into the forward call by monkey-patching (see below).
        pass

    def register(self, unet):
        """
        Register forward hooks on every cross-attention layer in the UNet
        **decoder** (up_blocks).  We capture the attention weight tensor
        produced for each token.
        """
        self._maps = []
        self._hooks = []

        def make_hook(layer_name: str):
            def hook(module, args, kwargs, output):
                # diffusers CrossAttention returns the hidden states as output;
                # We capture attention probs if available via the module attribute.
                if hasattr(module, "_attn_probs"):
                    self._maps.append((layer_name, module._attn_probs.detach()))
            return hook

        # Patch the processor to expose attention probs
        for name, module in unet.named_modules():
            if "up_blocks" in name and hasattr(module, "get_attention_scores"):
                h = module.register_forward_hook(make_hook(name), with_kwargs=True)
                self._hooks.append(h)

    def get_and_clear(self):
        maps = list(self._maps)
        self._maps = []
        return maps

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ---------------------------------------------------------------------------
# Attention map extraction via processor monkey-patch
# ---------------------------------------------------------------------------

class AttnProbeProcessor:
    """
    Custom attention processor that stores the cross-attention weight map
    and then delegates to the standard scaled-dot-product attention.

    Compatible with diffusers ≥ 0.20 AttnProcessor2_0 API.
    """

    def __init__(self, store: list):
        self.store = store

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        import torch.nn.functional as F

        residual = hidden_states
        hidden_states = attn.group_norm(hidden_states) if hasattr(attn, "group_norm") and attn.group_norm else hidden_states

        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        key = attn.to_k(kv_input)
        value = attn.to_v(kv_input)

        # Reshape to (batch * heads, seq, head_dim)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_probs = attn.get_attention_scores(query, key, attention_mask)

        # Only store cross-attention maps (encoder_hidden_states is not None)
        if is_cross:
            self.store.append(attn_probs.detach())

        hidden_states = torch.bmm(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# Collect decoder cross-attention maps for [V*] token
# ---------------------------------------------------------------------------

def extract_vstar_attention_map(
    unet,
    latent_size: int,
    vstar_token_index: int,
    attn_store: list,
) -> torch.Tensor:
    """
    Given the accumulated cross-attention maps in `attn_store` (collected
    during a UNet forward pass on the *object* branch), extract and average
    the maps for the [V*] token, then resize to latent_size × latent_size.

    Returns
    -------
    A_vstar : (B, 1, latent_size, latent_size) tensor averaged over decoder heads.
    """
    maps = []
    for attn_w in attn_store:
        # attn_w shape: (batch * heads, spatial_tokens, text_tokens)
        bh, spatial, text_len = attn_w.shape
        # [V*] token map
        if vstar_token_index >= text_len:
            continue
        vstar_map = attn_w[:, :, vstar_token_index]  # (bh, spatial)

        # Infer spatial resolution
        side = int(spatial ** 0.5)
        if side * side != spatial:
            continue  # skip non-square feature maps

        # Reshape to (bh, 1, side, side)
        vstar_map = vstar_map.view(bh, 1, side, side)

        # Resize to target latent size
        vstar_map = F.interpolate(
            vstar_map.float(),
            size=(latent_size, latent_size),
            mode="bilinear",
            align_corners=False,
        )
        maps.append(vstar_map)

    if not maps:
        return None

    # Average across decoder layers (and implicitly over heads since bh includes them)
    avg_map = torch.stack(maps, dim=0).mean(0)  # (bh, 1, H, W)

    # Average over heads: first dim is batch*heads, reshape to (B, heads, 1, H, W)
    # We don't know the number of heads directly, so we take the full mean
    avg_map = avg_map.mean(0, keepdim=True)  # (1, 1, H, W) — simplified

    # Normalise to [0, 1] for comparison with the binary mask
    _min = avg_map.amin(dim=(-2, -1), keepdim=True)
    _max = avg_map.amax(dim=(-2, -1), keepdim=True)
    avg_map = (avg_map - _min) / (_max - _min + 1e-8)

    return avg_map  # (1, 1, latent_H, latent_W)


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
