"""
DefectFill Dataset

Directory layout expected under `data_root`:
    data_root/
        defective/          ← defect images  (RGB, any common format)
        defective_masks/    ← pixel-accurate binary masks (white = defect region)
        good/               ← defect-free images

The defect and good image pairs are matched by filename stem where possible;
otherwise pairs are formed randomly each epoch.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def _list_images(folder: str) -> List[Path]:
    return sorted(
        p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS
    )


def _load_rgb(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return img


def _load_mask(path: Path, size: int) -> Image.Image:
    """Return a single-channel mask PIL image (0 / 255)."""
    mask = Image.open(path).convert("L")
    mask = mask.resize((size, size), Image.NEAREST)
    # Binarise: anything > 127 is defect (foreground = 1)
    mask = mask.point(lambda p: 255 if p > 127 else 0)
    return mask


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL RGB image → float tensor in [-1, 1]."""
    return TF.to_tensor(img) * 2.0 - 1.0


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """Convert single-channel PIL mask → float tensor in {0, 1}."""
    return TF.to_tensor(mask)  # values 0.0 or 1.0


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def joint_resize_crop(
    image: Image.Image,
    mask: Image.Image,
    target_size: int,
    resize_min: float = 1.0,
    resize_max: float = 1.125,
) -> Tuple[Image.Image, Image.Image]:
    """
    Randomly upscale image+mask by a factor in [resize_min, resize_max],
    then random-crop back to target_size × target_size.
    """
    scale = random.uniform(resize_min, resize_max)
    new_size = int(target_size * scale)

    image = image.resize((new_size, new_size), Image.LANCZOS)
    mask = mask.resize((new_size, new_size), Image.NEAREST)

    # Random crop
    i, j, h, w = T.RandomCrop.get_params(image, (target_size, target_size))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

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
    Generate M_rand: a binary mask composed of *num_boxes* axis-aligned
    rectangles whose side lengths are sampled uniformly in
    [min_frac * size, max_frac * size].

    Returns a (1, H, W) float tensor with values in {0, 1}.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    for _ in range(num_boxes):
        # Independent widths and heights
        bh = random.randint(
            max(1, int(min_frac * height)),
            max(1, int(max_frac * height)),
        )
        bw = random.randint(
            max(1, int(min_frac * width)),
            max(1, int(max_frac * width)),
        )
        y0 = random.randint(0, max(0, height - bh))
        x0 = random.randint(0, max(0, width - bw))
        mask[y0 : y0 + bh, x0 : x0 + bw] = 1.0

    return torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class DefectFillDataset(Dataset):
    """
    Returns one training sample per reference defect image.

    Each sample is a dict:
        image       : (3, H, W) float tensor in [-1, 1]   — the defect image I
        mask        : (1, H, W) float tensor in {0, 1}    — defect mask M
        mask_rand   : (1, H, W) float tensor in {0, 1}    — random box mask M_rand
        b_def       : (3, H, W) float tensor in [-1, 1]   — background B_def = (1-M)⊙I
        b_rand      : (3, H, W) float tensor in [-1, 1]   — background B_rand = (1-M_rand)⊙I
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 512,
        resize_min: float = 1.0,
        resize_max: float = 1.125,
        num_rand_boxes: int = 30,
        rand_box_min_frac: float = 0.03,
        rand_box_max_frac: float = 0.25,
        augment: bool = True,
    ):
        self.image_size = image_size
        self.resize_min = resize_min
        self.resize_max = resize_max
        self.num_rand_boxes = num_rand_boxes
        self.rand_box_min_frac = rand_box_min_frac
        self.rand_box_max_frac = rand_box_max_frac
        self.augment = augment

        defective_dir = os.path.join(data_root, "defective")
        masks_dir = os.path.join(data_root, "defective_masks")

        defective_paths = _list_images(defective_dir)
        mask_paths = _list_images(masks_dir)

        # Match by stem; fall back to positional pairing
        mask_by_stem = {p.stem: p for p in mask_paths}
        self.pairs: List[Tuple[Path, Path]] = []
        for dp in defective_paths:
            mp = mask_by_stem.get(dp.stem)
            if mp is None and mask_paths:
                # Try fuzzy: mask stem starts with image stem
                candidates = [m for m in mask_paths if m.stem.startswith(dp.stem)]
                mp = candidates[0] if candidates else None
            if mp is not None:
                self.pairs.append((dp, mp))

        if not self.pairs:
            raise FileNotFoundError(
                f"No matched (image, mask) pairs found under '{data_root}'. "
                "Ensure 'defective/' and 'defective_masks/' contain files "
                "with matching stems."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.pairs[idx]

        image = _load_rgb(img_path, self.image_size)
        mask_pil = _load_mask(mask_path, self.image_size)

        # Joint augmentation
        if self.augment:
            image, mask_pil = joint_resize_crop(
                image, mask_pil, self.image_size, self.resize_min, self.resize_max
            )

        img_t = _to_tensor(image)                   # (3,H,W) ∈ [-1,1]
        mask_t = _mask_to_tensor(mask_pil)           # (1,H,W) ∈ {0,1}

        # B_def = (1 - M) ⊙ I  (mask region zeroed out)
        b_def = img_t * (1.0 - mask_t)

        # M_rand and B_rand
        mask_rand = generate_random_box_mask(
            self.image_size,
            self.image_size,
            self.num_rand_boxes,
            self.rand_box_min_frac,
            self.rand_box_max_frac,
        )
        b_rand = img_t * (1.0 - mask_rand)

        return {
            "image": img_t,
            "mask": mask_t,
            "mask_rand": mask_rand,
            "b_def": b_def,
            "b_rand": b_rand,
        }


# ---------------------------------------------------------------------------
# Good-images dataset (used during inference/generation)
# ---------------------------------------------------------------------------

class GoodImagesDataset(Dataset):
    """Simple dataset for defect-free images used during inference."""

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
