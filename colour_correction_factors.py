#!/usr/bin/env python3
"""
Script 06b: Generate per-image colour correction factors from ColorChecker cards
=================================================================================
Processes a folder of JPG images, detects the ColorChecker Classic card in each,
and outputs a CSV of correction factors (slope + intercept per Lab channel).

You then apply these to your data separately in R or Excel using:
    L_corrected = L_slope * L_raw + L_intercept
    a_corrected = a_slope * a_raw + a_intercept
    b_corrected = b_slope * b_raw + b_intercept

Usage:
    python colour_correction_factors.py --images /path/to/jpgs --output correction_factors.csv

Requirements:
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


# =============================================================================
# CONFIGURATION
# =============================================================================
CHECKER_REFERENCE = "ColorChecker24 - After November 2014"
IMAGE_EXTENSIONS  = [".jpg", ".JPG", ".jpeg", ".JPEG", ".tif", ".TIF"]

# Exclude patches below this detected L* — these are clipped/underexposed
# and give unreliable correction estimates
MIN_L_DETECTED = 8
MIN_L_REF      = 20
MIN_PATCHES    = 10   # minimum usable patches; skip image if fewer found

# dE above this after correction means the fit itself failed (bad patch
# detection, wrong grid orientation, glare, motion blur) rather than
# ordinary camera/lighting noise. Anything above this is "poor" regardless
# of R^2, and should be excluded from correction application downstream,
# not just flagged.
MAX_DE_AFTER   = 10


# =============================================================================
# BUILD REFERENCE Lab VALUES (the "ground truth" the card should match)
# =============================================================================
# sRGB's native reference white is D65, and colour.sRGB_to_XYZ() always
# returns XYZ expressed under D65 regardless of what illuminant the pixels
# started under. Lab must be computed against that same D65 white, or the
# result isn't standard Lab under either illuminant. This used to pass D50
# here, which silently produced non-standard Lab (see chat for the size of
# that error — mean +11 units on b*, up to +18, with a* and L* barely
# touched). colour.XYZ_to_Lab()'s default illuminant is already D65, so
# leaving the argument out (rather than hardcoding it) is the safest fix.
def build_reference_lab():
    cc_ref = colour.CCS_COLOURCHECKERS[CHECKER_REFERENCE]
    ref_lab = []
    for name in list(cc_ref.data.keys()):
        XYZ = colour.xyY_to_XYZ(cc_ref.data[name])
        rgb = np.clip(colour.XYZ_to_sRGB(XYZ, illuminant=cc_ref.illuminant), 0, 1)
        ref_lab.append(colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb)))
    return np.array(ref_lab)


# =============================================================================
# CONVERT DETECTED sRGB PATCHES TO Lab
# =============================================================================
def srgb_to_lab(srgb_array):
    return np.array([
        colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(rgb, 0, 1)))
        for rgb in srgb_array
    ])


# =============================================================================
# DETECT CHECKER — tries all 4 rotations
# =============================================================================
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


# =============================================================================
# FIT CORRECTION — tries all 8 grid orientations, returns best
# =============================================================================
def fit_correction(detected_lab, ref_lab):
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

    best = {"mean_r2": -999, "corr": None, "orient": "", "n": 0}

    for label, cand in candidates:
        mask = (cand[:, 0] > MIN_L_DETECTED) & (ref_lab[:, 0] > MIN_L_REF)
        n = mask.sum()
        if n < MIN_PATCHES:
            continue

        det_m = cand[mask]
        ref_m = ref_lab[mask]
        corr  = {}

        for i, ch in enumerate(["L", "a", "b"]):
            x = det_m[:, i].reshape(-1, 1)
            y = ref_m[:, i]
            X = np.hstack([x, np.ones_like(x)])
            c, _, _, _ = lstsq(X, y, rcond=None)
            pred   = X @ c
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            corr[ch] = {
                "slope":     round(float(c[0]), 5),
                "intercept": round(float(c[1]), 5),
                "r2":        round(float(1 - ss_res / ss_tot)
                                   if ss_tot > 0 else 0, 4)
            }

        mean_r2 = np.mean([corr[c]["r2"] for c in ["L", "a", "b"]])
        if mean_r2 > best["mean_r2"]:
            best = {"mean_r2": mean_r2, "corr": corr,
                    "orient": label, "n": n, "cand": cand}

    return best


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate per-image colour correction factors from ColorChecker cards"
    )
    parser.add_argument("--images", "-i", required=True,
                        help="Folder containing JPG images")
    parser.add_argument("--output", "-o", default="correction_factors.csv",
                        help="Output CSV filename (default: correction_factors.csv)")
    args = parser.parse_args()

    image_dir = Path(args.images)
    images    = sorted([f for f in image_dir.rglob("*")
                        if f.suffix in IMAGE_EXTENSIONS])

    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(images)} images in {image_dir}")
    print(f"Output will be saved to: {args.output}\n")

    ref_lab = build_reference_lab()
    rows    = []

    n_good = 0; n_acceptable = 0; n_failed = 0

    for i, img_path in enumerate(images):
        if i % 50 == 0 and i > 0:
            print(f"  [{i}/{len(images)}]  "
                  f"good={n_good}  acceptable={n_acceptable}  failed={n_failed}")

        # Load image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            rows.append({"image_id": img_path.stem, "status": "load_failed"})
            n_failed += 1
            continue

        # Detect checker (tries all 4 rotations)
        det_srgb, rot_label = detect_patches(img_bgr)

        if det_srgb is None:
            rows.append({"image_id": img_path.stem,
                         "status":   "no_checker_found",
                         "note":     "use day-level fallback"})
            n_failed += 1
            continue

        # Convert to Lab and fit correction
        det_lab = srgb_to_lab(det_srgb)
        best    = fit_correction(det_lab, ref_lab)

        if best["corr"] is None:
            rows.append({"image_id": img_path.stem, "status": "fit_failed"})
            n_failed += 1
            continue

        corr = best["corr"]
        cand = best["cand"]

        # Delta-E before and after (quality check)
        dE_before = float(np.sqrt(((det_lab - ref_lab)**2).sum(axis=1)).mean())
        corr_lab  = np.column_stack([
            corr["L"]["slope"] * cand[:, 0] + corr["L"]["intercept"],
            corr["a"]["slope"] * cand[:, 1] + corr["a"]["intercept"],
            corr["b"]["slope"] * cand[:, 2] + corr["b"]["intercept"],
        ])
        dE_after = float(np.sqrt(((corr_lab - ref_lab)**2).sum(axis=1)).mean())

        quality = ("good"       if best["mean_r2"] > 0.85 and dE_after < 8
                   else "acceptable" if best["mean_r2"] > 0.70 and dE_after < MAX_DE_AFTER
                   else "poor")

        rows.append({
            "image_id":    img_path.stem,
            "status":      "calibrated",
            "quality":     quality,
            "rotation":    rot_label,
            "orientation": best["orient"],
            "n_patches":   best["n"],
            "L_slope":     corr["L"]["slope"],
            "L_intercept": corr["L"]["intercept"],
            "L_r2":        corr["L"]["r2"],
            "a_slope":     corr["a"]["slope"],
            "a_intercept": corr["a"]["intercept"],
            "a_r2":        corr["a"]["r2"],
            "b_slope":     corr["b"]["slope"],
            "b_intercept": corr["b"]["intercept"],
            "b_r2":        corr["b"]["r2"],
            "dE_before":   round(dE_before, 3),
            "dE_after":    round(dE_after,  3),
        })

        if quality == "good":             n_good       += 1
        elif quality == "acceptable":     n_acceptable += 1
        else:                             n_failed     += 1

    # Save report
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    # Print summary
    print(f"\n{'='*50}")
    print(f"DONE — {len(images)} images processed")
    print(f"  Good        (R²>0.85, dE<8):  {n_good}")
    print(f"  Acceptable  (R²>0.70):         {n_acceptable}")
    print(f"  Failed:                         {n_failed}")
    print(f"\nCorrection factors saved to: {args.output}")
    print(f"\nTo apply in R:")
    print(f"  report <- read.csv('{args.output}')")
    print(f"  df <- left_join(df, report, by = c('Image.ID' = 'image_id'))")
    print(f"  df <- df |> mutate(")
    print(f"    L_cal = L_slope * Lightness + L_intercept,")
    print(f"    a_cal = a_slope * A          + a_intercept,")
    print(f"    b_cal = b_slope * B          + b_intercept)")

if __name__ == "__main__":
    main()
