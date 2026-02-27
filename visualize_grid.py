"""
DefectFill Grid Search Visualiser
==================================
Reads output/grid_search/ and produces a single PNG grid:
  rows  = checkpoints  (step_1000, step_1500, final)
  cols  = CFG values   (3.0, 5.0, 7.5)
  cells = first N_SHOW defect images from that run

Usage:
    python visualise_grid.py
    python visualise_grid.py --grid_root output/grid_search --n_show 3 --out grid.png
"""

import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINTS = ["step_1000", "step_1500", "final"]
CFGS        = ["1.0", "1.5", "2.5", "3.0", "3.5", "5.0", "7.5", "10"]
DEFECT      = "crack"          # prefix used in folder names

IMG_SIZE    = 256              # thumbnail size per cell
PADDING     = 8
LABEL_H     = 28               # height of label bar above each column / row
FONT_SIZE   = 16


def find_images(folder: Path, n: int) -> list:
    """Return up to n image paths from folder, skipping mask files."""
    exts = {".png", ".jpg", ".jpeg"}
    imgs = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in exts and "_mask_" not in p.name
    )
    return imgs[:n]


def make_label(text: str, w: int, h: int, bg=(40, 40, 40), fg=(240, 240, 240)) -> Image.Image:
    img = Image.new("RGB", (w, h), color=bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (h - th) // 2), text, fill=fg, font=font)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_root", default="/mimer/NOBACKUP/groups/cast_fm/axel/Models/DefectGen1_output/generated/grid_search")
    parser.add_argument("--defect",    default=DEFECT)
    parser.add_argument("--n_show",    type=int, default=2,
                        help="Images per grid cell (default 2)")
    parser.add_argument("--out",       default="grid_search_results.png")
    args = parser.parse_args()

    root    = Path(args.grid_root)
    n_show  = args.n_show
    n_ckpts = len(CHECKPOINTS)
    n_cfgs  = len(CFGS)

    cell_w = IMG_SIZE
    cell_h = IMG_SIZE

    # Total canvas dimensions
    # +1 col for row labels, +1 row for column labels
    total_w = LABEL_H + (cell_w + PADDING) * n_cfgs  * n_show + PADDING
    total_h = LABEL_H + (cell_h + PADDING) * n_ckpts + PADDING
    canvas  = Image.new("RGB", (total_w, total_h), color=(20, 20, 20))

    # --- Column headers (CFG values) ---
    for ci, cfg in enumerate(CFGS):
        for si in range(n_show):
            x = LABEL_H + PADDING + (ci * n_show + si) * (cell_w + PADDING)
            label = make_label(f"CFG {cfg}" if si == 0 else "", cell_w, LABEL_H)
            canvas.paste(label, (x, 0))

    # --- Row labels + images ---
    for ri, ckpt in enumerate(CHECKPOINTS):
        y_base = LABEL_H + PADDING + ri * (cell_h + PADDING)

        # Row label
        row_label = make_label(ckpt.replace("step_", "step "), LABEL_H, cell_h,
                               bg=(60, 60, 80))
        canvas.paste(row_label, (0, y_base))

        for ci, cfg in enumerate(CFGS):
            folder_name = f"{args.defect}_{ckpt}_cfg{cfg}"
            folder      = root / folder_name

            if not folder.exists():
                # Grey placeholder
                ph = Image.new("RGB", (cell_w, cell_h), color=(80, 80, 80))
                draw = ImageDraw.Draw(ph)
                draw.text((8, cell_h // 2), "missing", fill=(200, 200, 200))
                for si in range(n_show):
                    x = LABEL_H + PADDING + (ci * n_show + si) * (cell_w + PADDING)
                    canvas.paste(ph, (x, y_base))
                continue

            imgs = find_images(folder, n_show)

            for si in range(n_show):
                x = LABEL_H + PADDING + (ci * n_show + si) * (cell_w + PADDING)
                if si < len(imgs):
                    img = Image.open(imgs[si]).convert("RGB").resize(
                        (cell_w, cell_h), Image.LANCZOS
                    )
                else:
                    img = Image.new("RGB", (cell_w, cell_h), color=(50, 50, 50))
                canvas.paste(img, (x, y_base))

    canvas.save(args.out)
    print(f"Saved: {args.out}  ({canvas.width}×{canvas.height} px)")


if __name__ == "__main__":
    main()