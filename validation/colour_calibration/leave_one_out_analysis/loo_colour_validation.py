#!/usr/bin/env python3
"""
Leave-one-out cross-validation for the colour correction fit
==============================================================
Companion script to 01_colour_correction_factors.py.

For each image, the ColorChecker card is detected and the same 4x6/6x4
orientation search is used to select the best patch grid. The per-channel
linear correction (L*, a*, b*) is then evaluated two ways:

  in_sample  -- the correction is fit on ALL detected patches and
                scored on those same patches (this is what dE_after
                in correction_factors.csv reports).

  loo        -- leave-one-out cross-validation: for each patch in turn,
                the correction is refit using every OTHER patch, and
                the held-out patch is then predicted from that fit.
                This is scored on data the fit never saw, so it is a
                more honest estimate of how the correction performs
                on a new colour it was not trained on.

Error is reported two ways:
  - overall dE   : the standard CIE76 Euclidean distance in Lab space,
                   combining all three channels into one number.
  - per-channel  : mean absolute error separately for L*, a*, and b*,
                   so you can see which channel drives the error (e.g.
                   whether lightness is systematically worse than the
                   colour channels, as is typical for this correction).

Output: one row per image in a summary CSV (in-sample vs held-out, both
overall dE and per-channel MAE), plus an optional long-format CSV with
per-patch results for every channel.

Usage:
    python loo_colour_validation.py \
        --images path/to/images \
        --output loo_validation_summary.csv \
        --patch-output loo_validation_per_patch.csv

Install the required packages if not already:
    pip install colour-science colour-checker-detection opencv-python numpy pandas
"""

import cv2
import numpy as np
import pandas as pd
import colour
import colour_checker_detection
import argparse
from pathlib import Path
from numpy.linalg import lstsq

# CONFIGURATION -- kept identical to 01_colour_correction_factors.py so the
# patch detection and orientation selection match exactly.
CHECKER_REFERENCE = "ColorChecker24 - After November 2014"
IMAGE_EXTENSIONS  = [".jpg", ".JPG", ".jpeg", ".JPEG", ".tif", ".TIF"]

MIN_L_DETECTED = 8
MIN_L_REF      = 20
MIN_PATCHES    = 10

CHANNEL_NAMES = ["L", "a", "b"]


# BUILD REFERENCE Lab VALUES FROM COLOUR CHECKER
def build_reference_lab():
    cc_ref = colour.CCS_COLOURCHECKERS[CHECKER_REFERENCE]
    ref_lab = []
    for name in list(cc_ref.data.keys()):
        XYZ = colour.xyY_to_XYZ(cc_ref.data[name])
        rgb = np.clip(colour.XYZ_to_sRGB(XYZ, illuminant=cc_ref.illuminant), 0, 1)
        ref_lab.append(colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb)))
    return np.array(ref_lab), list(cc_ref.data.keys())


# CONVERT DETECTED sRGB PATCHES TO Lab
def srgb_to_lab(srgb_array):
    return np.array([
        colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(rgb, 0, 1)))
        for rgb in srgb_array
    ])


# DETECT COLOUR CHECKER (tries all four rotations, as in script 01)
def detect_patches(img_bgr):
    rotations = [
        ("0°",     img_bgr),
        ("90°CW",  cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)),
        ("180°",   cv2.rotate(img_bgr, cv2.ROTATE_180)),
        ("90°CCW", cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]
    for rot_label, img_r in rotations:
        img_f = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB) / 255.0
        try:
            result = colour_checker_detection.detect_colour_checkers_segmentation(
                img_f, additional_data=True)
            if result:
                return result[0].swatch_colours, rot_label
        except Exception:
            continue
    return None, "failed"


# SELECT BEST ORIENTATION (same 8-way search as script 01, but here we only
# need the winning patch grid + mask; we refit ourselves below for LOO).
def select_orientation(detected_lab, ref_lab):
    g46 = detected_lab.reshape(4, 6, 3)
    g64 = detected_lab.reshape(6, 4, 3)
    candidates = [
        ("4x6",        g46.reshape(24, 3)),
        ("4x6_flipLR", g46[:, ::-1].reshape(24, 3)),
        ("4x6_flipUD", g46[::-1].reshape(24, 3)),
        ("4x6_rot180", g46[::-1, ::-1].reshape(24, 3)),
        ("6x4",        g64.reshape(24, 3)),
        ("6x4_flipLR", g64[:, ::-1].reshape(24, 3)),
        ("6x4_flipUD", g64[::-1].reshape(24, 3)),
        ("6x4_rot180", g64[::-1, ::-1].reshape(24, 3)),
    ]

    best = {"mean_r2": -999, "orient": "", "n": 0, "cand": None, "mask": None}

    for label, cand in candidates:
        mask = (cand[:, 0] > MIN_L_DETECTED) & (ref_lab[:, 0] > MIN_L_REF)
        n = mask.sum()
        if n < MIN_PATCHES:
            continue

        det_m = cand[mask]
        ref_m = ref_lab[mask]
        r2s = []
        for i in range(3):
            x = det_m[:, i].reshape(-1, 1)
            y = ref_m[:, i]
            X = np.hstack([x, np.ones_like(x)])
            c, _, _, _ = lstsq(X, y, rcond=None)
            pred = X @ c
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)

        mean_r2 = np.mean(r2s)
        if mean_r2 > best["mean_r2"]:
            best = {"mean_r2": mean_r2, "orient": label, "n": n,
                     "cand": cand, "mask": mask}

    return best


def fit_channel(x, y):
    """Fit y = slope*x + intercept, return [slope, intercept]."""
    X = np.hstack([x.reshape(-1, 1), np.ones((len(x), 1))])
    c, _, _, _ = lstsq(X, y, rcond=None)
    return c


def in_sample_errors(det_m, ref_m):
    """Fit on all patches, predict all patches (mirrors dE_after logic,
    restricted to the same patch subset used for fitting).

    Returns:
        residuals: (n_patches, 3) array of signed (pred - ref) per channel
        dE:        (n_patches,) overall CIE76 Euclidean distance
    """
    preds = np.zeros_like(det_m)
    for ch in range(3):
        c = fit_channel(det_m[:, ch], ref_m[:, ch])
        preds[:, ch] = c[0] * det_m[:, ch] + c[1]
    residuals = preds - ref_m
    dE = np.sqrt((residuals ** 2).sum(axis=1))
    return residuals, dE


def loo_errors(det_m, ref_m):
    """Leave-one-out CV: refit n times, holding out one patch each time,
    and score the held-out patch only.

    Returns:
        residuals: (n_patches, 3) array of signed (pred - ref) per channel
        dE:        (n_patches,) overall CIE76 Euclidean distance
    """
    n = det_m.shape[0]
    residuals = np.zeros((n, 3))
    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i])
        for ch in range(3):
            c = fit_channel(det_m[train_idx, ch], ref_m[train_idx, ch])
            pred = c[0] * det_m[i, ch] + c[1]
            residuals[i, ch] = pred - ref_m[i, ch]
    dE = np.sqrt((residuals ** 2).sum(axis=1))
    return residuals, dE


def mae_per_channel(residuals):
    """Mean absolute error per channel from an (n_patches, 3) residual array."""
    return np.abs(residuals).mean(axis=0)


# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Leave-one-out cross-validation of the colour correction fit"
    )
    parser.add_argument("--images", "-i", required=True,
                         help="Folder containing JPG/TIF images")
    parser.add_argument("--output", "-o", default="loo_validation_summary.csv",
                         help="Output CSV: one row per image (default: loo_validation_summary.csv)")
    parser.add_argument("--patch-output", "-p", default=None,
                         help="Optional output CSV: one row per patch, long format")
    args = parser.parse_args()

    image_dir = Path(args.images)
    images = sorted([f for f in image_dir.rglob("*")
                      if f.suffix in IMAGE_EXTENSIONS])

    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(images)} images in {image_dir}")

    ref_lab, patch_names = build_reference_lab()
    summary_rows = []
    patch_rows = []

    n_ok = 0
    n_skipped = 0

    for i, img_path in enumerate(images):
        if i % 50 == 0 and i > 0:
            print(f"  [{i}/{len(images)}]  ok={n_ok}  skipped={n_skipped}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            summary_rows.append({"image_id": img_path.stem, "status": "load_failed"})
            n_skipped += 1
            continue

        det_srgb, rot_label = detect_patches(img_bgr)
        if det_srgb is None:
            summary_rows.append({"image_id": img_path.stem,
                                  "status": "no_checker_found"})
            n_skipped += 1
            continue

        det_lab = srgb_to_lab(det_srgb)
        best = select_orientation(det_lab, ref_lab)

        if best["cand"] is None:
            summary_rows.append({"image_id": img_path.stem, "status": "fit_failed"})
            n_skipped += 1
            continue

        mask = best["mask"]
        det_m = best["cand"][mask]
        ref_m = ref_lab[mask]
        names_m = [n for n, keep in zip(patch_names, mask) if keep]

        in_res, in_dE = in_sample_errors(det_m, ref_m)
        loo_res, loo_dE = loo_errors(det_m, ref_m)

        in_mae = mae_per_channel(in_res)     # [L_mae, a_mae, b_mae]
        loo_mae = mae_per_channel(loo_res)   # [L_mae, a_mae, b_mae]

        row = {
            "image_id":    img_path.stem,
            "status":      "ok",
            "rotation":    rot_label,
            "orientation": best["orient"],
            "n_patches":   int(mask.sum()),

            # overall CIE76 dE (all three channels combined)
            "in_sample_dE": round(float(in_dE.mean()), 3),
            "loo_dE":        round(float(loo_dE.mean()), 3),
            "dE_inflation_pct": round(
                float(100 * (loo_dE.mean() - in_dE.mean()) / in_dE.mean()), 1
            ) if in_dE.mean() > 0 else None,
        }

        # per-channel mean absolute error, in-sample and held-out
        for ch_i, ch in enumerate(CHANNEL_NAMES):
            row[f"in_sample_{ch}_MAE"] = round(float(in_mae[ch_i]), 3)
            row[f"loo_{ch}_MAE"]       = round(float(loo_mae[ch_i]), 3)
            row[f"{ch}_MAE_inflation_pct"] = round(
                float(100 * (loo_mae[ch_i] - in_mae[ch_i]) / in_mae[ch_i]), 1
            ) if in_mae[ch_i] > 0 else None

        summary_rows.append(row)
        n_ok += 1

        if args.patch_output:
            for j, name in enumerate(names_m):
                patch_rows.append({
                    "image_id":     img_path.stem,
                    "patch_name":   name,
                    "ref_L":        round(float(ref_m[j, 0]), 3),
                    "in_sample_dE": round(float(in_dE[j]), 3),
                    "loo_dE":       round(float(loo_dE[j]), 3),
                    "in_sample_L_error": round(float(in_res[j, 0]), 3),
                    "in_sample_a_error": round(float(in_res[j, 1]), 3),
                    "in_sample_b_error": round(float(in_res[j, 2]), 3),
                    "loo_L_error":       round(float(loo_res[j, 0]), 3),
                    "loo_a_error":       round(float(loo_res[j, 1]), 3),
                    "loo_b_error":       round(float(loo_res[j, 2]), 3),
                })

    df = pd.DataFrame(summary_rows)
    df.to_csv(args.output, index=False)

    print(f"\n{'='*50}")
    print(f"DONE — {len(images)} images processed")
    print(f"  Successfully validated: {n_ok}")
    print(f"  Skipped (no card / load / fit failure): {n_skipped}")
    if n_ok > 0:
        ok_df = df[df["status"] == "ok"]
        print(f"\n  Overall dE (all channels combined):")
        print(f"    in-sample mean: {ok_df['in_sample_dE'].mean():.3f}")
        print(f"    held-out mean:  {ok_df['loo_dE'].mean():.3f}")
        print(f"\n  Per-channel mean absolute error (in-sample -> held-out):")
        for ch in CHANNEL_NAMES:
            print(f"    {ch}*:  {ok_df[f'in_sample_{ch}_MAE'].mean():.3f}  ->  "
                  f"{ok_df[f'loo_{ch}_MAE'].mean():.3f}")
    print(f"\nSummary saved to: {args.output}")

    if args.patch_output:
        pd.DataFrame(patch_rows).to_csv(args.patch_output, index=False)
        print(f"Per-patch detail saved to: {args.patch_output}")


if __name__ == "__main__":
    main()
