"""
calculate_iou.py
----------------
Compares manually segmented abalone lip images (segmented in Photoshop) against Python-segmented
versions by calculating Intersection over Union (IoU) for each matched pair.

Folder structure expected:
    root_dir/
        manual/       PNG files, RGBA, transparent background, no suffix
        python/       PNG files, RGB, white background, filename ends in _lip

Usage:
python calculate_iou.py --root "path\to\root" --manual_dir "path\to\manually\segmented\images" --python_dir "path\to\python\segmented\images"

"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def make_mask_manual(path: Path, alpha_thresh: int = 127) -> np.ndarray:
    """Binary mask from RGBA image: foreground = alpha > alpha_thresh."""
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = np.array(img)[:, :, 3]
    return alpha > alpha_thresh


def make_mask_python(path: Path, white_thresh: float = 10.0) -> np.ndarray:
    """Binary mask from RGB image: foreground = distance from pure white > white_thresh."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img).astype(np.float32)
    dist = np.sqrt(((arr - 255.0) ** 2).sum(axis=2))
    return dist > white_thresh


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> dict:
    """Return IoU and component counts for two boolean masks of the same shape."""
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"Mask shapes do not match: {mask_a.shape} vs {mask_b.shape}"
        )
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    iou = float(intersection) / float(union) if union > 0 else float("nan")
    return {
        "pixels_manual": int(mask_a.sum()),
        "pixels_python": int(mask_b.sum()),
        "intersection": int(intersection),
        "union": int(union),
        "iou": iou,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate IoU for lip segmentation pairs.")
    parser.add_argument("--root",         required=True, help="Path to parent folder containing the two subfolders.")
    parser.add_argument("--manual_dir",   default="manual", help="Name of the manual segmentation subfolder (default: manual).")
    parser.add_argument("--python_dir",   default="python",  help="Name of the Python segmentation subfolder (default: python).")
    parser.add_argument("--alpha_thresh", type=int,   default=127,  help="Alpha threshold for manual mask (default: 127).")
    parser.add_argument("--white_thresh", type=float, default=10.0, help="Distance-from-white threshold for Python mask (default: 10).")
    parser.add_argument("--output",       default="iou_results.csv", help="Output CSV filename (default: iou_results.csv).")
    args = parser.parse_args()

    root       = Path(args.root)
    manual_dir = root / args.manual_dir
    python_dir = root / args.python_dir

    for d in (manual_dir, python_dir):
        if not d.is_dir():
            sys.exit(f"ERROR: Directory not found: {d}")

    # Build lookup: stem -> path for Python files (strip _lip suffix)
    python_files = {
        p.stem.removesuffix("_lip"): p
        for p in sorted(python_dir.glob("*.png"))
    }

    manual_files = sorted(manual_dir.glob("*.png"))

    if not manual_files:
        sys.exit(f"ERROR: No PNG files found in {manual_dir}")

    results = []
    unmatched = []

    for manual_path in manual_files:
        stem = manual_path.stem  # e.g. IMG_1803
        if stem not in python_files:
            unmatched.append(manual_path.name)
            continue

        python_path = python_files[stem]

        try:
            mask_m = make_mask_manual(manual_path, args.alpha_thresh)
            mask_p = make_mask_python(python_path, args.white_thresh)
            metrics = compute_iou(mask_m, mask_p)
        except Exception as e:
            print(f"  WARNING: Could not process {stem}: {e}")
            results.append({
                "filename": stem,
                "pixels_manual": "ERROR",
                "pixels_python": "ERROR",
                "intersection": "ERROR",
                "union": "ERROR",
                "iou": "ERROR",
            })
            continue

        results.append({"filename": stem, **metrics})
        print(f"  {stem:40s}  IoU = {metrics['iou']:.4f}  "
              f"(manual {metrics['pixels_manual']:,} px, python {metrics['pixels_python']:,} px)")

    # Write CSV
    out_path = root / args.output
    fieldnames = ["filename", "pixels_manual", "pixels_python", "intersection", "union", "iou"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    valid_ious = [r["iou"] for r in results if isinstance(r["iou"], float) and not np.isnan(r["iou"])]
    print()
    print("=" * 60)
    print(f"Pairs processed:   {len(results)}")
    print(f"Unmatched manual:  {len(unmatched)}")
    if unmatched:
        for name in unmatched:
            print(f"   {name}")
    if valid_ious:
        print(f"Mean IoU:          {np.mean(valid_ious):.4f}")
        print(f"Median IoU:        {np.median(valid_ious):.4f}")
        print(f"Min IoU:           {np.min(valid_ious):.4f}")
        print(f"Max IoU:           {np.max(valid_ious):.4f}")
    print(f"Results saved to:  {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()