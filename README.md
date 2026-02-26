# DefectFill

Realistic defect image generation via fine-tuned inpainting diffusion models.

> Implementation of **DefectFill: Realistic Defect Generation with Inpainting Diffusion Model for Visual Inspection** (CVPR 2024).

---

## Overview

DefectFill fine-tunes a Stable Diffusion 2 inpainting model with LoRA to learn a specific defect concept from a small set of reference images. Three complementary loss terms drive training:

| Loss | Weight | Purpose |
|------|--------|---------|
| **Defect loss** `L_def` | 0.5 | Precisely captures intrinsic defect features |
| **Object loss** `L_obj` | 0.2 | Learns the semantic relationship between defect and object |
| **Attention loss** `L_attn` | 0.05 | Ensures [V*] token attends to the defect region |

After training, **Low-Fidelity Selection (LFS)** generates 8 candidates per (image, mask) pair and selects the one with the highest LPIPS score inside the masked region — the most "realistic" defect.

---

## Repository Structure

```
defectfill/
├── config.py         # TrainingConfig and GenerationConfig dataclasses
├── dataset.py        # DefectFillDataset, GoodImagesDataset, augmentations, random masks
├── losses.py         # L_def, L_obj, L_attn, DefectFill combined loss
├── utils.py          # LoRA helpers, LFS, image I/O, LR scheduler
├── train.py          # Training entry point
├── generate.py       # Generation entry point with LFS
└── requirements.txt
```

---

## Dataset Layout

Two layouts are supported and auto-detected at runtime.

**Flat** (original):
```
data/<object>/
    defective/          ← defect images
    defective_masks/    ← binary masks
    good/               ← defect-free images
```

**Nested** (train/test wrapper + defect-type subfolder):
```
data/<object>/
    train/
        defective/
            crack/      ← defect images
        defective_masks/
            crack/      ← binary masks
    test/
        good/           ← defect-free images
```

Image and mask files must share the same filename stem within their respective
sub-folders (e.g. `crack/001.png` ↔ `crack/001.png`).

Pass the **object root** (`data/concrete`) to both `train.py` and `generate.py`;
`train/`, `test/`, `good/`, and defect-type sub-folders are resolved automatically.

### Train / test split

The code mirrors the paper's protocol: all defect image–mask pairs (regardless of
which sub-folder they live in) are pooled, then split **1/3 train / 2/3 test**
with a fixed random seed. With 21 pairs this gives 7 training pairs and 14 test
masks applied to good images during generation.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0, CUDA recommended.

---

## Training

```bash
python train.py \
    --data_root data/hazelnut \
    --object_name hazelnut \
    --output_dir output/hazelnut_hole \
    --train_steps 2000 \
    --batch_size 4 \
    --unet_lr 2e-4 \
    --text_encoder_lr 4e-5 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lambda_def 0.5 \
    --lambda_obj 0.2 \
    --lambda_attn 0.05 \
    --alpha 0.3 \
    --seed 42
```

Key training details (from paper):
- **Base model**: `sd2-community/stable-diffusion-2-inpainting`
- **LoRA** on UNet attention layers + text encoder projection matrices
- **Warmup**: linear 0 → LR over first 100 steps
- **Augmentation**: random resize ×[1.0, 1.125] + centre crop
- **Random masks** M_rand: 30 boxes, sides 3–25% of image size
- **[V*] token**: the word `sks`

---

## Generation

```bash
python generate.py \
    --pretrained_model_name sd2-community/stable-diffusion-2-inpainting \
    --lora_weights_path output/hazelnut_hole/unet_lora_final \
    --te_lora_weights_path output/hazelnut_hole/text_encoder_lora_final \
    --good_images_dir data/hazelnut/good \
    --masks_dir data/hazelnut/generation_masks \
    --output_dir generated/hazelnut_hole \
    --object_name hazelnut \
    --num_samples_lfs 8 \
    --num_inference_steps 50 \
    --guidance_scale 7.5
```

For each good image × mask pair, 8 samples are generated and the one with the highest LPIPS score inside the mask is saved.

---

## Method Details

### Defect Loss (Eq. 5)

```
L_def = E[ || M ⊙ (ε − ε_θ(x_t^def, t, c^def)) ||² ]
```

Background image: `B_def = (1 − M) ⊙ I`  
Input: `x_t^def = concat(x_t, b_def, M)`  
Prompt `P_def = "A photo of sks"`

### Object Loss (Eq. 7)

```
L_obj = E[ || M' ⊙ (ε − ε_θ(x_t^obj, t, c^obj)) ||² ]
M' = M + α·(1 − M),   α = 0.3
```

Random box mask M_rand (30 boxes), `B_rand = (1 − M_rand) ⊙ I`  
Input: `x_t^obj = concat(x_t, b_rand, M_rand)`  
Prompt `P_obj = "A <object> with sks"`

### Attention Loss (Eq. 8)

```
L_attn = E[ || A_t^[V*] − M ||² ]
```

Cross-attention maps from UNet **decoder** (up_blocks) only, averaged over layers and resized to latent resolution.

### Combined Loss (Eq. 9)

```
L_ours = 0.5·L_def + 0.2·L_obj + 0.05·L_attn
```

---

## Citation

```bibtex
@inproceedings{song2024defectfill,
  title={DefectFill: Realistic Defect Generation with Inpainting Diffusion Model for Visual Inspection},
  author={Song, Jaewoo and Park, Daemin and Baek, Kanghyun and Lee, Sangyub and Choi, Jooyoung and Kim, Eunji and Yoon, Sungroh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```