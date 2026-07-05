#!/usr/bin/env python3
"""
extract_lip_colour.py
----------------------
Run immediately after segment_lips.py. Takes the white-background lip cutouts
in <root>/segmented/, applies the same HSB colour-threshold used in
Whole_Color_Macro_All_components_FINAL_components.ijm to exclude background
and over/under-exposed pixels, then extracts mean RGB, HSB and CIELAB values
over the remaining pixels.

Nothing under --root/segmented or --out/polygons is touched or overwritten.
Note: unlike the Macro, this script does NOT save over the input image — the
Macro's "Clear Outside" + saveAs step overwrote the file it had just opened,
which isn't repeated here on purpose.

Outputs (all new, written under --root):
  * <root>/Whole_Color_Measurements_pivoted.xlsx
        Sheet 'colour_data'    - same column layout as the Macro's pivoted
                                  CSV: image_ID, mean red, green, blue, hue,
                                  saturation, brightness, lightness, a, b
        Sheet 'extraction_log' - per-image status (ok / no_pixels_after_threshold
                                  / unreadable), for QC
  * <root>/colour_threshold_qc/<mirrors segmented/ subfolders>/<name>_thresh.jpg
        QC image: green tint = pixels used for the colour means, red tint =
        lip pixels excluded by the threshold (too bright/dark/grey), white
        background left as-is.

Threshold boundaries are copied from the Macro (see THRESHOLD below) — change
them there if you recalibrate.

IMPORTANT — Lab conversion pipeline:
CIELAB here is computed with the same sRGB(D65)->XYZ->Lab(D65) pipeline as
the corrected colour_correction_factors.py, NOT ImageJ's native Lab Stack.
This is deliberate, not an oversight: uncorrected_L/a/b must be computed the
same way the correction factors were fitted for collate_colour_data.py's
slope/intercept correction to be numerically valid. As of the D65 fix, this
pipeline now produces standard CIE Lab, so it also happens to match what a
correct ImageJ Lab Stack conversion would give you (small residual
differences are just implementation/rounding, not a white-point mismatch).
RGB and HSB have no equivalent white-point ambiguity, so they're computed
with the standard formula (same maths ImageJ's HSB Stack uses).

Usage:
    python extract_lip_colour.py --root "C:\\Users\\RebeccaPedler\\Documents\\lip_cutouts_test"

Requirements:
    pip install opencv-python numpy pandas openpyxl matplotlib colour-science
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import colour
from matplotlib.colors import rgb_to_hsv

# =============================================================================
# THRESHOLD BOUNDARIES — copied from
# Whole_Color_Macro_All_components_FINAL_components.ijm (lines ~57-65).
# Channel order matches the Macro: Hue, Saturation, Brightness, all on the
# 0-255 scale ImageJ's "HSB Stack" uses. filter="pass" keeps values inside
# [min, max]. The Macro also supports filter="stop" (invert) but doesn't use
# it for this image set, so inversion isn't implemented here.
# =============================================================================
THRESHOLD = {
    "hue":        {"min": 25, "max": 255},
    "saturation": {"min": 0,  "max": 255},
    "brightness": {"min": 0,  "max": 160},
}

LIP_SUFFIX = "_lip"
IMG_EXT = {".jpg", ".jpeg", ".png"}
COLOUR_DATA_COLUMNS = [
    "image_ID", "mean red", "green", "blue",
    "hue", "saturation", "brightness", "lightness", "a", "b",
]


def image_id_from_cutout(path: Path) -> str:
    """IMG_2076_lip.jpg -> IMG_2076, so this lines up with the image_id/stem
    used in correction_factors.csv and segment_lips.py's summary.csv."""
    stem = path.stem
    if stem.lower().endswith(LIP_SUFFIX):
        stem = stem[: -len(LIP_SUFFIX)]
    return stem


def rgb_to_lab(rgb_pixels):
    """rgb_pixels: (N, 3) array, values in [0, 1]. Returns (N, 3) Lab array
    (standard sRGB->XYZ->Lab, D65 throughout — matches colour_correction_factors.py's
    srgb_to_lab())."""
    XYZ = colour.sRGB_to_XYZ(np.clip(rgb_pixels, 0, 1))
    return colour.XYZ_to_Lab(XYZ)


def find_cutouts(root: Path):
    seg_dir = root / "segmented"
    if not seg_dir.exists():
        raise FileNotFoundError(
            f"No 'segmented' folder under {root}. Point --root at the same "
            f"--out folder you gave segment_lips.py."
        )
    return sorted(p for p in seg_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXT)


def threshold_and_measure(bgr):
    """Returns (means_dict_or_None, mask, status)."""
    rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_f = rgb_u8.astype(np.float64) / 255.0

    hsv = rgb_to_hsv(rgb_f) * 255.0  # match ImageJ's 0-255 HSB Stack scale
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    t = THRESHOLD
    mask = ((h >= t["hue"]["min"])        & (h <= t["hue"]["max"]) &
            (s >= t["saturation"]["min"]) & (s <= t["saturation"]["max"]) &
            (v >= t["brightness"]["min"]) & (v <= t["brightness"]["max"]))

    if not mask.any():
        return None, mask, "no_pixels_after_threshold"

    rgb_kept = rgb_u8[mask].astype(np.float64)
    hsv_kept = hsv[mask]
    lab_kept = rgb_to_lab(rgb_kept / 255.0)

    means = {
        "mean red":   rgb_kept[:, 0].mean(),
        "green":      rgb_kept[:, 1].mean(),
        "blue":       rgb_kept[:, 2].mean(),
        "hue":        hsv_kept[:, 0].mean(),
        "saturation": hsv_kept[:, 1].mean(),
        "brightness": hsv_kept[:, 2].mean(),
        "lightness":  lab_kept[:, 0].mean(),
        "a":          lab_kept[:, 1].mean(),
        "b":          lab_kept[:, 2].mean(),
    }
    return means, mask, "ok"


def save_threshold_qc(bgr, mask, out_path: Path):
    """Green = pixels used for the colour means. Red = pixels that are part
    of the lip cutout (i.e. not white background) but got excluded by the
    threshold. White background is left untouched, same visual language as
    segment_lips.py's overlay (green = kept)."""
    lip_region = ~np.all(bgr >= 250, axis=-1)  # near-white = background
    excluded_in_lip = lip_region & ~mask

    qc = bgr.copy()
    qc[mask] = (0.4 * qc[mask] + np.array([0, 180, 0]) * 0.6).astype(np.uint8)
    qc[excluded_in_lip] = (0.4 * qc[excluded_in_lip]
                            + np.array([0, 0, 220]) * 0.6).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), qc, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="the --out folder you gave segment_lips.py "
                         "(must contain a 'segmented' subfolder)")
    ap.add_argument("--output-name", default="Whole_Color_Measurements_pivoted.xlsx",
                    help="Excel filename, written directly under --root")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    cutouts = find_cutouts(root)
    if not cutouts:
        print(f"No lip cutouts (*_lip.jpg/.png) found under {root / 'segmented'}")
        return

    qc_root = root / "colour_threshold_qc"
    print(f"Found {len(cutouts)} lip cutout(s) under {root / 'segmented'}\n")

    rows, log_rows = [], []
    n_ok = n_empty = n_bad = 0

    for i, path in enumerate(cutouts, 1):
        image_id = image_id_from_cutout(path)
        bgr = cv2.imread(str(path))

        if bgr is None:
            log_rows.append([image_id, str(path), "unreadable"])
            n_bad += 1
            print(f"[{i}/{len(cutouts)}] {path.name}: unreadable")
            continue

        means, mask, status = threshold_and_measure(bgr)

        rel = path.relative_to(root / "segmented")
        save_threshold_qc(bgr, mask, qc_root / rel.parent / f"{path.stem}_thresh.jpg")

        if status == "no_pixels_after_threshold":
            log_rows.append([image_id, str(path), status])
            n_empty += 1
            print(f"[{i}/{len(cutouts)}] {path.name}: no pixels passed the threshold")
            continue

        rows.append({"image_ID": image_id, **means})
        log_rows.append([image_id, str(path), status])
        n_ok += 1
        print(f"[{i}/{len(cutouts)}] {path.name}: ok "
              f"(L={means['lightness']:.1f} a={means['a']:.1f} b={means['b']:.1f})")

    colour_df = pd.DataFrame(rows, columns=COLOUR_DATA_COLUMNS).round(3)
    log_df = pd.DataFrame(log_rows, columns=["image_ID", "path", "status"])

    out_xlsx = root / args.output_name
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        colour_df.to_excel(writer, sheet_name="colour_data", index=False)
        log_df.to_excel(writer, sheet_name="extraction_log", index=False)

    print(f"\n{'='*50}")
    print(f"DONE — {len(cutouts)} cutout(s) processed")
    print(f"  ok:                        {n_ok}")
    print(f"  no pixels after threshold: {n_empty}")
    print(f"  unreadable:                {n_bad}")
    print(f"\nColour data:  {out_xlsx}")
    print(f"QC overlays:  {qc_root}")


if __name__ == "__main__":
    main()
