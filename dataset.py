"""
DefectFill Dataset
==================

Directory layout expected under ``data_root``:

    data_root/
        defective/          ← defect images  (RGB, any common format)
        defective_masks/    ← pixel-accurate binary masks (white = defect region)
        good/               ← defect-free images

Train / test split
------------------
Following the paper, pairs are split deterministically into
  • 1/3 for training  (fine-tuning the diffusion model)
  • 2/3 for testing   (good images whose masks are used during generation)

With 21 pairs the split is 7 train / 14 test.

Thin-mask-safe resizing
-----------------------
``Image.NEAREST`` hard-snaps pixels to 0/255 during resize; a feature that
is only 1–2 pixels wide can disappear entirely.  Instead we:

  1. Resize with ``Image.BILINEAR``  → sub-pixel thin strokes become low-
     probability floating-point values (they are no longer exactly 0).
  2. Re-binarise with a *very low* threshold (> 0.5/255 ≈ 0.002 in [0,1])
     — any pixel that received even a trace of signal from a thin feature
     is preserved as "defect".

This keeps the binary nature of the mask while being robust to thin defect
boundaries such as cracks or scratches.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# Any pixel with normalised value above this threshold is kept as "defect".
# 0.002 ≈ 0.5/255 → preserves anything that bilinear smearing made non-zero.
MASK_BINARISE_THRESHOLD = 0.002


# ---------------------------------------------------------------------------
# Low-level I/O helpers
# ---------------------------------------------------------------------------

def _list_images(folder: str) -> List[Path]:
    return sorted(
        p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS
    )


def _load_rgb(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return img


def _load_mask(path: Path, size: int) -> Image.Image:
    """
    Load a defect mask and resize it to ``size × size`` safely for thin features.

    Strategy
    --------
    1. Open as greyscale, binarise the source image first (threshold 127) so
       any prior JPEG artefacts don't pollute the signal.
    2. Resize with BILINEAR — thin features become low but non-zero.
    3. Re-binarise: pixel > MASK_BINARISE_THRESHOLD → 255, else 0.

    Returns a single-channel PIL image with values in {0, 255}.
    """
    mask = Image.open(path).convert("L")

    # Step 1: clean-binarise the source to remove JPEG/compression noise
    src_arr = np.array(mask, dtype=np.float32) / 255.0
    src_arr = (src_arr > 0.5).astype(np.float32)
    mask = Image.fromarray((src_arr * 255).astype(np.uint8), mode="L")

    # Step 2: bilinear resize (thin-safe)
    mask = mask.resize((size, size), Image.BILINEAR)

    # Step 3: re-binarise with low threshold
    mask_arr = np.array(mask, dtype=np.float32) / 255.0
    binary = (mask_arr > MASK_BINARISE_THRESHOLD).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB → (3, H, W) float tensor in [-1, 1]."""
    return TF.to_tensor(img) * 2.0 - 1.0


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """Single-channel PIL mask → (1, H, W) float tensor in {0.0, 1.0}."""
    return TF.to_tensor(mask)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_pairs(
    pairs: List[Tuple[Path, Path]],
    train_fraction: float = 1.0 / 3.0,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Deterministically split matched (image, mask) pairs into train and test sets.

    The paper trains on 1/3 of the defect pairs and uses the remaining 2/3 of
    masks (applied to good images) for generation and evaluation.

    Parameters
    ----------
    pairs           : full list of (defective_image_path, mask_path) tuples
    train_fraction  : fraction assigned to training (default 1/3)
    seed            : RNG seed for reproducibility

    Returns
    -------
    train_pairs, test_pairs
    """
    n = len(pairs)
    n_train = max(1, round(n * train_fraction))

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    train_idx = set(indices[:n_train])
    train_pairs = [pairs[i] for i in range(n) if i in train_idx]
    test_pairs  = [pairs[i] for i in range(n) if i not in train_idx]

    return train_pairs, test_pairs


# ---------------------------------------------------------------------------
# Joint resize + random crop augmentation
# ---------------------------------------------------------------------------

def joint_resize_crop(
    image: Image.Image,
    mask: Image.Image,
    target_size: int,
    resize_min: float = 1.0,
    resize_max: float = 1.125,
) -> Tuple[Image.Image, Image.Image]:
    """
    Randomly upscale image + mask by a factor in [resize_min, resize_max],
    then apply a *random* crop back to target_size × target_size.

    Why random crop (not centre crop)?
    -----------------------------------
    The paper says the image and mask are "randomly resized … and then cropped
    back to their original size."  A *random* crop (rather than a fixed centre
    crop) is the standard interpretation: it varies which sub-region ends up in
    the training sample, giving the model diversity in the spatial position of
    the defect and preventing it from learning a location prior.  A centre crop
    would always extract the same region and provide no positional augmentation.

    The mask uses the thin-safe bilinear + re-binarise strategy during the
    upscale step so fine cracks/scratches survive the resize.
    """
    scale = random.uniform(resize_min, resize_max)
    new_size = int(target_size * scale)

    # --- Upscale image (LANCZOS for quality) ---
    image = image.resize((new_size, new_size), Image.LANCZOS)

    # --- Upscale mask (thin-safe bilinear → re-binarise) ---
    mask = mask.resize((new_size, new_size), Image.BILINEAR)
    mask_arr = np.array(mask, dtype=np.float32) / 255.0
    binary = (mask_arr > MASK_BINARISE_THRESHOLD).astype(np.uint8) * 255
    mask = Image.fromarray(binary, mode="L")

    # --- Same random crop applied to both ---
    i, j, h, w = T.RandomCrop.get_params(image, (target_size, target_size))
    image = TF.crop(image, i, j, h, w)
    mask  = TF.crop(mask,  i, j, h, w)

    return image, mask


# ---------------------------------------------------------------------------
# Random box mask (M_rand)
# ---------------------------------------------------------------------------

def generate_random_box_mask(
    height: int,
    width: int,
    num_boxes: int = 30,
    min_frac: float = 0.03,
    max_frac: float = 0.25,
) -> torch.Tensor:
    """
    Generate M_rand: a binary mask of *num_boxes* axis-aligned rectangles
    whose side lengths are sampled uniformly in [min_frac·size, max_frac·size].

    Returns a (1, H, W) float tensor with values in {0.0, 1.0}.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    for _ in range(num_boxes):
        bh = random.randint(
            max(1, int(min_frac * height)),
            max(1, int(max_frac * height)),
        )
        bw = random.randint(
            max(1, int(min_frac * width)),
            max(1, int(max_frac * width)),
        )
        y0 = random.randint(0, max(0, height - bh))
        x0 = random.randint(0, max(0, width  - bw))
        mask[y0 : y0 + bh, x0 : x0 + bw] = 1.0

    return torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)


# ---------------------------------------------------------------------------
# Training Dataset
# ---------------------------------------------------------------------------

class DefectFillDataset(Dataset):
    """
    Dataset for fine-tuning the inpainting diffusion model.

    Pass ``split="train"`` to get the training third, ``split="test"`` for the
    evaluation two-thirds, or ``split="all"`` for every pair.

    Each sample is a dict:
        image       : (3, H, W) float in [-1, 1]   — defect image I
        mask        : (1, H, W) float in {0, 1}    — defect mask M
        mask_rand   : (1, H, W) float in {0, 1}    — random box mask M_rand
        b_def       : (3, H, W) float in [-1, 1]   — B_def = (1−M) ⊙ I
        b_rand      : (3, H, W) float in [-1, 1]   — B_rand = (1−M_rand) ⊙ I
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        train_fraction: float = 1.0 / 3.0,
        split_seed: int = 42,
        image_size: int = 512,
        resize_min: float = 1.0,
        resize_max: float = 1.125,
        num_rand_boxes: int = 30,
        rand_box_min_frac: float = 0.03,
        rand_box_max_frac: float = 0.25,
        augment: bool = True,
    ):
        assert split in ("train", "test", "all"), \
            "split must be 'train', 'test', or 'all'"

        self.image_size      = image_size
        self.resize_min      = resize_min
        self.resize_max      = resize_max
        self.num_rand_boxes  = num_rand_boxes
        self.rand_box_min_frac = rand_box_min_frac
        self.rand_box_max_frac = rand_box_max_frac
        self.augment         = augment

        # ---- Collect all matched (defective, mask) pairs ----
        defective_dir = os.path.join(data_root, "defective")
        masks_dir     = os.path.join(data_root, "defective_masks")

        defective_paths = _list_images(defective_dir)
        mask_paths      = _list_images(masks_dir)

        mask_by_stem = {p.stem: p for p in mask_paths}
        all_pairs: List[Tuple[Path, Path]] = []
        for dp in defective_paths:
            mp = mask_by_stem.get(dp.stem)
            if mp is None:
                # Fuzzy fallback: mask stem starts with image stem
                candidates = [m for m in mask_paths if m.stem.startswith(dp.stem)]
                mp = candidates[0] if candidates else None
            if mp is not None:
                all_pairs.append((dp, mp))

        if not all_pairs:
            raise FileNotFoundError(
                f"No matched (image, mask) pairs found under '{data_root}'. "
                "Ensure 'defective/' and 'defective_masks/' have files with "
                "matching stems."
            )

        # ---- Deterministic train / test split ----
        train_pairs, test_pairs = split_pairs(
            all_pairs, train_fraction=train_fraction, seed=split_seed
        )

        if split == "train":
            self.pairs = train_pairs
        elif split == "test":
            self.pairs = test_pairs
        else:
            self.pairs = all_pairs

        print(
            f"[DefectFillDataset] split='{split}' | "
            f"train={len(train_pairs)} | test={len(test_pairs)} | "
            f"using={len(self.pairs)} pairs"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.pairs[idx]

        image    = _load_rgb(img_path, self.image_size)
        mask_pil = _load_mask(mask_path, self.image_size)

        # Joint augmentation: random resize + random crop
        if self.augment:
            image, mask_pil = joint_resize_crop(
                image, mask_pil,
                self.image_size,
                self.resize_min,
                self.resize_max,
            )

        img_t  = _to_tensor(image)          # (3, H, W) in [-1, 1]
        mask_t = _mask_to_tensor(mask_pil)  # (1, H, W) in {0, 1}

        # B_def = (1 − M) ⊙ I
        b_def = img_t * (1.0 - mask_t)

        # M_rand and B_rand for the object branch
        mask_rand = generate_random_box_mask(
            self.image_size, self.image_size,
            self.num_rand_boxes,
            self.rand_box_min_frac,
            self.rand_box_max_frac,
        )
        b_rand = img_t * (1.0 - mask_rand)

        return {
            "image":     img_t,
            "mask":      mask_t,
            "mask_rand": mask_rand,
            "b_def":     b_def,
            "b_rand":    b_rand,
        }


# ---------------------------------------------------------------------------
# Good-images dataset (used during inference / generation)
# ---------------------------------------------------------------------------

class GoodImagesDataset(Dataset):
    """Simple dataset for defect-free images used at inference time."""

    def __init__(self, good_dir: str, image_size: int = 512):
        self.paths = _list_images(good_dir)
        self.image_size = image_size
        if not self.paths:
            raise FileNotFoundError(f"No images found in '{good_dir}'")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = _load_rgb(self.paths[idx], self.image_size)
        return _to_tensor(img), str(self.paths[idx])