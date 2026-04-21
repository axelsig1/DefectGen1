"""
DefectFill Training Script
==========================

Fine-tunes a Stable Diffusion 2 inpainting model with LoRA to learn
the concept of a specific defect type from a small set of reference images.

Usage
-----
python train.py \
    --data_root data/hazelnut \
    --object_name hazelnut \
    --output_dir output/hazelnut_hole \
    [--train_steps 2000] \
    [--batch_size 4] \
    [--seed 42]

Or pass a JSON / dataclass config (see config.py).
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# For validation loss
import lpips
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
import torchvision.transforms.functional as TF


from config import TrainingConfig
from dataset import DefectFillDataset, GoodImagesDataset
from losses import defectfill_loss, AttnProbeProcessor
from torch.utils.tensorboard import SummaryWriter   # Tensorboard for loss plots
from utils import (
    set_seed,
    make_defect_prompt,
    make_object_prompt,
    get_token_index,
    add_lora_to_unet,
    add_lora_to_text_encoder,
    get_linear_warmup_scheduler,
    compute_lpips_in_mask,      # val loss
    tensor_to_pil,              # val loss
    pil_to_tensor,              # val loss
)

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing (thin wrapper around TrainingConfig)
# ---------------------------------------------------------------------------

def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train DefectFill")
    # Data
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_fraction", type=float, default=1.0/3.0, 
                        help="Fraction of dataset to use for training (default 1/3)")
    parser.add_argument("--object_name", type=str, required=True)
    parser.add_argument("--defect_type", type=str, default=None,
                        help="Defect subfolder, e.g. crack. Omit to use all types.")
    parser.add_argument("--placeholder_token", type=str, default="sks")
    parser.add_argument("--image_size", type=int, default=512)
    # Training
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--unet_lr", type=float, default=2e-4)
    parser.add_argument("--text_encoder_lr", type=float, default=4e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=50)
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    # Loss weights
    parser.add_argument("--lambda_def", type=float, default=0.5)
    parser.add_argument("--lambda_obj", type=float, default=0.2)
    parser.add_argument("--lambda_attn", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.3)
    # Model
    parser.add_argument("--pretrained_model_name", type=str,
                        default="sd2-community/stable-diffusion-2-inpainting")

    args = parser.parse_args()
    cfg = TrainingConfig()
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_models(cfg: TrainingConfig):
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        StableDiffusionInpaintPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    logger.info(f"Loading pretrained model: {cfg.pretrained_model_name}")

    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_name, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.pretrained_model_name, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model_name, subfolder="unet"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg.pretrained_model_name, subfolder="scheduler"
    )

    return tokenizer, text_encoder, vae, unet, noise_scheduler


def log_validation(
    vae, text_encoder, tokenizer, unet, cfg, device, weight_dtype, 
    tb_writer, global_step, val_dataset, val_good_dataset, lpips_fn
):
    logger.info(f"Running validation at step {global_step}...")
    
    # 1. Freeze models temporarily
    unet.eval()
    text_encoder.eval()

    # 2. Build a temporary, fast pipeline using the currently loaded weights
    val_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        cfg.pretrained_model_name, subfolder="scheduler"
    )
    pipe = StableDiffusionInpaintPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=val_scheduler,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # 3. Grab the Ground Truth defect and mask
    val_sample = val_dataset[0] 
    gt_tensor = val_sample["image"].unsqueeze(0).to(device)   # Ground Truth defective image
    mask_tensor = val_sample["mask"].unsqueeze(0).to(device)  # Ground Truth mask
    
    # 4. Grab a totally clean background image to use as our canvas
    good_img_tensor, _ = val_good_dataset[0]
    good_tensor = good_img_tensor.unsqueeze(0).to(device)     # Clean metal surface
    
    prompt = make_object_prompt(cfg.object_name, cfg.placeholder_token)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(cfg.mixed_precision != "no")):
            # Convert CLEAN image and mask for pipeline
            img_pil = tensor_to_pil(good_tensor.squeeze(0).cpu())
            mask_pil = TF.to_pil_image(mask_tensor.squeeze(0).cpu())

            # Generate the crack onto the CLEAN image
            result_pil = pipe(
                prompt=prompt,
                image=img_pil,
                mask_image=mask_pil,
                height=cfg.image_size,
                width=cfg.image_size,
                num_inference_steps=20, # Fast inference
                guidance_scale=7.5,
                generator=generator,
                output_type="pil",
            ).images[0]

            result_tensor = pil_to_tensor(result_pil).unsqueeze(0).to(device)

            # Enforce background paste-back using the CLEAN background
            mask_3ch = mask_tensor.expand_as(result_tensor)
            final_composite = result_tensor * mask_3ch + good_tensor * (1.0 - mask_3ch)

            # 5. Calculate Masked LPIPS
            # To prevent LPIPS from measuring the difference between the clean canvas 
            # background and the Ground Truth background inside the bounding box, 
            # we temporarily paste the generated crack onto the Ground Truth background.
            
            lpips_eval_tensor = result_tensor * mask_3ch + gt_tensor * (1.0 - mask_3ch)

            # Now, the backgrounds are 100% identical. The bounding-box LPIPS will 
            # STRICTLY measure the perceptual difference of the crack itself.
            
            # Compare generated image (on clean background) vs Ground Truth (original defect)
            lpips_score = compute_lpips_in_mask(
                lpips_fn, lpips_eval_tensor, gt_tensor, mask_tensor
            )

    # 6. Log to TensorBoard
    tb_writer.add_scalar("Validation/Masked_LPIPS", lpips_score, global_step)
    
    # Convert from [-1, 1] to [0, 1] for TensorBoard image logging
    display_img = (final_composite.squeeze(0) + 1.0) / 2.0
    tb_writer.add_image("Validation/Generated_Defect", display_img, global_step)

    # 7. Unfreeze models and resume training
    unet.train()
    text_encoder.train()
    
    # Free up memory
    del pipe
    torch.cuda.empty_cache()
    
    logger.info(f"Validation complete. LPIPS: {lpips_score:.4f}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainingConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = (
        torch.float16 if cfg.mixed_precision == "fp16"
        else torch.bfloat16 if cfg.mixed_precision == "bf16"
        else torch.float32
    )

    # ------------------------------------------------------------------ #
    # Load models
    # ------------------------------------------------------------------ #
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_models(cfg)

    # VAE stays frozen, moved to device at reduced precision
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)

    # ------------------------------------------------------------------ #
    # Apply LoRA
    # ------------------------------------------------------------------ #
    logger.info("Applying LoRA to UNet and text encoder …")
    unet = add_lora_to_unet(
        unet,
        rank=cfg.lora_rank,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
    )
    text_encoder = add_lora_to_text_encoder(
        text_encoder,
        rank=cfg.lora_rank,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
    )

    # UNet and text encoder must stay in fp32 so that their trainable LoRA
    # parameters have fp32 gradients. GradScaler requires fp32 parameters —
    # it unscales fp32 gradients after the backward pass; if the params are
    # already fp16 it raises "Attempting to unscale FP16 gradients".
    # autocast() handles casting activations to fp16 during the forward pass,
    # so the computation is still fast; only the stored weights stay fp32.
    unet.to(device)
    text_encoder.to(device)

    # Gradient checkpointing is intentionally disabled.
    # It re-runs the forward pass during backward, which conflicts with
    # AttnProbeProcessor: the stored attention maps from the first forward
    # pass have different tensor shapes than those from the recomputed pass,
    # causing a CheckpointError. With an A40 (48 GB) and batch_size=4 fp16
    # (~20 GB used), there is ample VRAM and checkpointing is not needed.
    # unet.enable_gradient_checkpointing()  # <-- do not re-enable

    # ------------------------------------------------------------------ #
    # Dataset & DataLoader
    # ------------------------------------------------------------------ #
    logger.info(f"Loading dataset from: {cfg.data_root}")
    dataset = DefectFillDataset(
        data_root=cfg.data_root,
        train_fraction=cfg.train_fraction,
        split="train",
        defect_type=cfg.defect_type,
        image_size=cfg.image_size,
        resize_min=cfg.resize_min,
        resize_max=cfg.resize_max,
        num_rand_boxes=cfg.num_rand_boxes,
        rand_box_min_frac=cfg.rand_box_min_frac,
        rand_box_max_frac=cfg.rand_box_max_frac,
        augment=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    # ------------------------------------------------------------------ #
    # Optimiser (separate LRs for UNet and text encoder)
    # ------------------------------------------------------------------ #
    unet_params = [p for p in unet.parameters() if p.requires_grad]
    te_params = [p for p in text_encoder.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": unet_params, "lr": cfg.unet_lr},
            {"params": te_params, "lr": cfg.text_encoder_lr},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = get_linear_warmup_scheduler(optimizer, cfg.warmup_steps, cfg.train_steps)

    # ------------------------------------------------------------------ #
    # Build text embeddings (pre-computed once per step to save time)
    # ------------------------------------------------------------------ #
    p_def = make_defect_prompt(cfg.placeholder_token)
    p_obj = make_object_prompt(cfg.object_name, cfg.placeholder_token)

    vstar_idx_obj = get_token_index(tokenizer, cfg.placeholder_token, p_obj)
    logger.info(f"Defect prompt : '{p_def}'")
    logger.info(f"Object prompt : '{p_obj}' (V* at index {vstar_idx_obj})")

    # ------------------------------------------------------------------ #
    # Attention probe setup (decoder layers only)
    # ------------------------------------------------------------------ #
    # We install custom processors to capture cross-attention maps during
    # the object-branch forward pass.

    attn_store_obj: list = []

    def set_attn_processors(model, store):
        """Replace cross-attention processors in decoder (up_blocks) with probe."""
        for name, module in model.named_modules():
            if "up_blocks" in name and hasattr(module, "set_processor"):
                module.set_processor(AttnProbeProcessor(store))

    def reset_attn_processors(model):
        """Restore default attention processors."""
        from diffusers.models.attention_processor import AttnProcessor2_0
        for name, module in model.named_modules():
            if "up_blocks" in name and hasattr(module, "set_processor"):
                module.set_processor(AttnProcessor2_0())

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    global_step = 0
    epoch = 0
    progress_bar = tqdm(total=cfg.train_steps, desc="Training")



    # Initialize TensorBoard Writer
    tb_writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, "logs"))

    # --- Setup Validation ---
    logger.info("Initializing Validation metrics...")
    val_lpips_fn = lpips.LPIPS(net="alex").to(device)
    val_lpips_fn.requires_grad_(False)
    
    # Load the test split for validation (no augmentation!)
    val_dataset = DefectFillDataset(
        data_root=cfg.data_root,
        train_fraction=cfg.train_fraction,
        split="test",
        defect_type=cfg.defect_type,
        image_size=cfg.image_size,
        augment=False 
    )

    # NEW: Load the good/clean images to act as our blank canvas
    # (dataset.py auto-resolves cfg.data_root to find the 'test/good' folder)
    val_good_dataset = GoodImagesDataset(
        good_dir=cfg.data_root, 
        image_size=cfg.image_size
    )



    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.mixed_precision == "fp16"))

    unet.train()
    text_encoder.train()

    while global_step < cfg.train_steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= cfg.train_steps:
                break

            # Move batch to device in fp32.
            # The UNet and text encoder are fp32; autocast casts activations
            # to fp16 internally during the forward pass.
            image = batch["image"].to(device)       # (B,3,H,W)
            mask = batch["mask"].to(device)          # (B,1,H,W)
            mask_rand = batch["mask_rand"].to(device)
            b_def = batch["b_def"].to(device)
            b_rand = batch["b_rand"].to(device)

            B = image.shape[0]

            with torch.amp.autocast('cuda', enabled=(cfg.mixed_precision != "no")):

                # --- Sample timestep and noise (shared between branches) ---
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()
                # VAE encoding: cast inputs to fp16 for the frozen VAE only.
                # Outputs are cast back to fp32 so the rest of the graph stays fp32.
                x0        = encode_latent(vae, image, weight_dtype).float()
                b_def_lat = encode_latent(vae, b_def,  weight_dtype).float()
                b_rand_lat= encode_latent(vae, b_rand, weight_dtype).float()

                latent_h, latent_w = x0.shape[-2], x0.shape[-1]

                noise = torch.randn_like(x0)   # fp32, same shape as x0

                # Noisy latent at sampled timestep (fp32)
                x_t = noise_scheduler.add_noise(x0, noise, timesteps)

                # Resize masks to latent resolution (fp32)
                mask_lat = F.interpolate(
                    mask.float(), size=(latent_h, latent_w), mode="nearest"
                )
                mask_rand_lat = F.interpolate(
                    mask_rand.float(), size=(latent_h, latent_w), mode="nearest"
                )

                # ---- Defect branch: x_t^def = concat(x_t, M, b_def) ----
                # diffusers SD2-inpainting UNet expects 9 channels in this order:
                #   [noisy_latents(4), mask(1), masked_image(4)]
                # Our earlier code had [x_t, b_def_lat, mask_lat] which is
                # [4, 4, 1] — mask and masked_image swapped. This caused the UNet
                # to receive image colours where it expected a binary mask and
                # vice versa, completely scrambling the learned representations.
                x_t_def = torch.cat([x_t, mask_lat, b_def_lat], dim=1)

                # Text embedding for P_def
                tokens_def = tokenizer(
                    [p_def] * B,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)
                c_def = text_encoder(tokens_def).last_hidden_state

                noise_pred_def = unet(
                    x_t_def, timesteps, encoder_hidden_states=c_def
                ).sample

                # ---- Object branch: x_t^obj = concat(x_t, M_rand, b_rand) ----
                # Same channel order fix: [noisy(4), mask(1), masked_image(4)]
                x_t_obj = torch.cat([x_t, mask_rand_lat, b_rand_lat], dim=1)

                tokens_obj = tokenizer(
                    [p_obj] * B,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)
                c_obj = text_encoder(tokens_obj).last_hidden_state

                # Install attention probes for the object branch
                attn_store_obj.clear()
                set_attn_processors(unet, attn_store_obj)

                noise_pred_obj = unet(
                    x_t_obj, timesteps, encoder_hidden_states=c_obj
                ).sample

                # Extract [V*] attention map from decoder layers.
                # Pass batch_size so the function can separate B from heads
                # and return (B, 1, H, W) instead of a collapsed (1, 1, H, W).
                attn_map_vstar = extract_decoder_attn_map(
                    attn_store_obj,
                    vstar_idx_obj,
                    latent_h,
                    latent_w,
                    device,
                    batch_size=B,
                )
                reset_attn_processors(unet)
                attn_store_obj.clear()

                # ---- Compute combined loss ----
                losses = defectfill_loss(
                    noise_pred_def=noise_pred_def,
                    noise_pred_obj=noise_pred_obj,
                    noise_target=noise,
                    mask=mask,
                    attn_map_vstar=attn_map_vstar,
                    alpha=cfg.alpha,
                    lambda_def=cfg.lambda_def,
                    lambda_obj=cfg.lambda_obj,
                    lambda_attn=cfg.lambda_attn,
                )
                loss = losses["loss"]

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                unet_params + te_params, cfg.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            progress_bar.update(1)

            if global_step % cfg.log_steps == 0:
                lr_unet = scheduler.get_last_lr()[0]
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    l_def=f"{losses['l_def'].item():.4f}",
                    l_obj=f"{losses['l_obj'].item():.4f}",
                    l_attn=f"{losses['l_attn'].item() if hasattr(losses['l_attn'], 'item') else losses['l_attn']:.4f}",
                    lr=f"{lr_unet:.2e}",
                )
                logger.info(
                    f"Step {global_step}: loss={loss.item():.4f} "
                    f"(def={losses['l_def'].item():.4f}, "
                    f"obj={losses['l_obj'].item():.4f}, "
                    f"attn={losses['l_attn'].item() if hasattr(losses['l_attn'], 'item') else 0.0:.4f})"
                )



                # Write to TensorBoard
                tb_writer.add_scalar("Loss/Total", loss.item(), global_step)
                tb_writer.add_scalar("Loss/Defect", losses['l_def'].item(), global_step)
                tb_writer.add_scalar("Loss/Object", losses['l_obj'].item(), global_step)
                
                # Handle l_attn which might be a float 0.0 if not computed
                attn_val = losses['l_attn'].item() if isinstance(losses['l_attn'], torch.Tensor) else losses['l_attn']
                tb_writer.add_scalar("Loss/Attention", attn_val, global_step)
                tb_writer.add_scalar("Learning_Rate", lr_unet, global_step)



            if global_step % cfg.save_steps == 0:
                save_checkpoint(unet, text_encoder, cfg.output_dir, global_step)
            
                # --- Run Live Validation ---
                log_validation(
                        vae, text_encoder, tokenizer, unet, cfg, device, weight_dtype, 
                        tb_writer, global_step, val_dataset, val_good_dataset, val_lpips_fn
                    )

    progress_bar.close()

    tb_writer.close()

    # Final save
    save_checkpoint(unet, text_encoder, cfg.output_dir, global_step, final=True)
    logger.info(f"Training complete. Weights saved to '{cfg.output_dir}'.")


# ---------------------------------------------------------------------------
# Helpers used in the training loop
# ---------------------------------------------------------------------------

def encode_latent(vae, images: torch.Tensor, weight_dtype) -> torch.Tensor:
    """Encode images to latent space without gradients."""
    with torch.no_grad():
        lat = vae.encode(images.to(dtype=weight_dtype)).latent_dist.sample()
        lat = lat * vae.config.scaling_factor
    return lat


def extract_decoder_attn_map(
    attn_store: list,
    vstar_token_idx: int,
    latent_h: int,
    latent_w: int,
    device,
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Average [V*] cross-attention maps across decoder layers.

    Returns (B, 1, latent_h, latent_w) — one map per image in the batch.

    Each element of attn_store has shape (B*heads, spatial, text_len).
    The previous implementation called .mean(0) which collapsed BOTH the
    batch AND head dimensions into a single (1, 1, H, W) map.  When that
    was compared against mask (B, 1, H, W) in the attention loss, the L2
    was computed between one smeared average and B distinct masks, causing
    the loss to fluctuate wildly and giving the model no useful gradient.

    Fix: reshape (B*heads, spatial) → (B, heads, spatial), average over
    heads only (dim=1), keeping the batch dimension intact.  Different
    decoder layers have different head counts so we infer heads = bh // B.
    """
    maps = []
    for attn_w in attn_store:
        bh, spatial, text_len = attn_w.shape
        if vstar_token_idx >= text_len:
            continue
        side = int(spatial ** 0.5)
        if side * side != spatial:
            continue
        if bh % batch_size != 0:
            continue   # skip if shapes are inconsistent

        heads = bh // batch_size

        token_map = attn_w[:, :, vstar_token_idx]              # (B*heads, spatial)
        token_map = token_map.view(batch_size, heads, side, side)  # (B, heads, side, side)

        # Average over heads only — keep batch dimension intact
        token_map = token_map.mean(dim=1, keepdim=True)        # (B, 1, side, side)

        token_map = F.interpolate(
            token_map.float(),
            size=(latent_h, latent_w),
            mode="bilinear",
            align_corners=False,
        )                                                       # (B, 1, latent_h, latent_w)
        maps.append(token_map)

    if not maps:
        return None

    # Stack across layers and average — shape stays (B, 1, latent_h, latent_w)
    avg = torch.stack(maps, dim=0).mean(0)

    # Normalise each image in the batch independently to [0, 1]
    _min = avg.amin(dim=(-2, -1), keepdim=True)
    _max = avg.amax(dim=(-2, -1), keepdim=True)
    avg = (avg - _min) / (_max - _min + 1e-8)

    return avg.to(device)


def save_checkpoint(unet, text_encoder, output_dir: str, step: int, final: bool = False):
    """Save LoRA weights for both models."""
    tag = "final" if final else f"step_{step}"
    unet_path = os.path.join(output_dir, f"unet_lora_{tag}")
    te_path = os.path.join(output_dir, f"text_encoder_lora_{tag}")
    unet.save_pretrained(unet_path)
    text_encoder.save_pretrained(te_path)
    logger.info(f"Checkpoint saved at step {step} → {output_dir}")


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)