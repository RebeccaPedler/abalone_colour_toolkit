#!/usr/bin/env python3
"""
collate_colour_data.py
-----------------------
Joins the outputs of colour_correction_factors.py, segment_lips.py and
extract_lip_colour.py into one Excel file, one row per image_id, and applies
the per-image Lab correction factors to the raw lip colour means.

Join key is the original photo's filename stem (e.g. "IMG_2076"):
  - correction_factors.csv                    -> image_id column, as-is
  - summary.csv (segment_lips.py)             -> stem of the 'image' path column
  - Whole_Color_Measurements_pivoted.xlsx     -> image_ID column, as-is
    (extract_lip_colour.py already strips the "_lip" suffix so these line up)

HSB and RGB are deliberately left out of this file — only CIELAB.

Output columns:
    image_id, lip_segmentation_status, lip_pct_of_frame,
    correction_status, correction_quality, correction_rotation,
    correction_orientation, n_patches,
    L_slope, L_intercept, L_r2, a_slope, a_intercept, a_r2,
    b_slope, b_intercept, b_r2, dE_before, dE_after,
    uncorrected_L, uncorrected_a, uncorrected_b,
    corrected_L, corrected_a, corrected_b

This is an outer join: an image missing from any one input still appears
(with blanks) rather than silently disappearing. Check the printed warning
and the blank cells before treating a row as final — a blank usually means
the checker wasn't detected in that image, or the lip segmentation failed.

Usage:
    python collate_colour_data.py \\
        --corrections correction_factors.csv \\
        --segmentation summary.csv \\
        --colour-data Whole_Color_Measurements_pivoted.xlsx \\
        --output collated_colour_data.xlsx

Requirements:
    pip install pandas openpyxl
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_COLUMN_ORDER = [
    "image_id", "lip_segmentation_status", "lip_pct_of_frame",
    "correction_status", "correction_quality", "correction_rotation",
    "correction_orientation", "n_patches",
    "L_slope", "L_intercept", "L_r2",
    "a_slope", "a_intercept", "a_r2",
    "b_slope", "b_intercept", "b_r2",
    "dE_before", "dE_after",
    "uncorrected_L", "uncorrected_a", "uncorrected_b",
    "corrected_L", "corrected_a", "corrected_b",
]


def stem_from_any_path(p: str) -> str:
    """Path(...).stem, but tolerant of Windows-style paths even if this
    script happens to run on a different OS than the one that wrote the CSV."""
    return Path(str(p).replace("\\", "/")).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrections", required=True, help="correction_factors.csv")
    ap.add_argument("--segmentation", required=True, help="summary.csv from segment_lips.py")
    ap.add_argument("--colour-data", required=True, help="Whole_Color_Measurements_pivoted.xlsx")
    ap.add_argument("--output", default="collated_colour_data.xlsx")
    args = ap.parse_args()

    corr = pd.read_csv(args.corrections)
    corr = corr.rename(columns={
        "status": "correction_status",
        "quality": "correction_quality",
        "rotation": "correction_rotation",
        "orientation": "correction_orientation",
    })

    seg = pd.read_csv(args.segmentation)
    seg["image_id"] = seg["image"].apply(stem_from_any_path)
    seg = seg.rename(columns={"status": "lip_segmentation_status"})
    seg = seg[["image_id", "lip_segmentation_status", "lip_pct_of_frame"]]

    colour = pd.read_excel(args.colour_data, sheet_name="colour_data")
    colour = colour.rename(columns={
        "image_ID":  "image_id",
        "lightness": "uncorrected_L",
        "a":         "uncorrected_a",
        "b":         "uncorrected_b",
    })[["image_id", "uncorrected_L", "uncorrected_a", "uncorrected_b"]]

    df = corr.merge(seg, on="image_id", how="outer") \
             .merge(colour, on="image_id", how="outer")

    have_factors = df[["L_slope", "a_slope", "b_slope"]].notna().all(axis=1) \
        & df[["uncorrected_L", "uncorrected_a", "uncorrected_b"]].notna().all(axis=1) \
        & (df["correction_quality"] != "poor")

    df["corrected_L"] = np.nan
    df["corrected_a"] = np.nan
    df["corrected_b"] = np.nan
    df.loc[have_factors, "corrected_L"] = (
        df.loc[have_factors, "L_slope"] * df.loc[have_factors, "uncorrected_L"]
        + df.loc[have_factors, "L_intercept"])
    df.loc[have_factors, "corrected_a"] = (
        df.loc[have_factors, "a_slope"] * df.loc[have_factors, "uncorrected_a"]
        + df.loc[have_factors, "a_intercept"])
    df.loc[have_factors, "corrected_b"] = (
        df.loc[have_factors, "b_slope"] * df.loc[have_factors, "uncorrected_b"]
        + df.loc[have_factors, "b_intercept"])

    col_order = [c for c in OUTPUT_COLUMN_ORDER if c in df.columns]
    extra_cols = [c for c in df.columns if c not in col_order]
    df = df[col_order + extra_cols].sort_values("image_id")

    n_missing = (~have_factors).sum()
    if n_missing:
        n_poor = (df["correction_quality"] == "poor").sum()
        print(f"Note: {n_missing}/{len(df)} image(s) have no corrected_L/a/b "
              f"({n_poor} excluded for 'poor' calibration quality, the rest "
              f"missing colour data or correction factors entirely). Check "
              f"lip_segmentation_status and correction_status/correction_quality "
              f"in {args.output} to see why.")

    df.round(3).to_excel(args.output, index=False)
    print(f"Collated {len(df)} image(s) -> {args.output}")


if __name__ == "__main__":
    main()
