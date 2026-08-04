#!/usr/bin/env python3
"""
Script 04: Collate all data into one csv and apply correction factors
=================================================================================
Joins the outputs of colour_correction_factors.py, segment_ROI.py and
extract_ROI_colour.py into one Excel file, one row per image_id, and applies
the per-image Lab correction factors to the raw ROI colour means.

Usage:
   python collate_colour_data.py --corrections correction_factors.csv --segmentation summary.csv 
   --colour-data whole_color_measurements.csv --output collated_colour_data.csv

Requirements:
    pip install pandas
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_COLUMN_ORDER = [
    "image_id", "ROI_segmentation_status", "ROI_pct_of_frame",
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
    ap.add_argument("--segmentation", required=True, help="summary.csv from segment_ROI.py")
    ap.add_argument("--colour-data", required=True, help="whole_color_measurements.csv")
    ap.add_argument("--output", default="collated_colour_data.csv")
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
    seg = seg.rename(columns={"status": "ROI_segmentation_status"})
    seg = seg[["image_id", "ROI_segmentation_status", "ROI_pct_of_frame"]]

    colour = pd.read_csv(args.colour_data)
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
              f"ROI_segmentation_status and correction_status/correction_quality "
              f"in {args.output} to see why.")

    df.round(3).to_csv(args.output, index=False)
    print(f"Collated {len(df)} image(s) -> {args.output}")


if __name__ == "__main__":
    main()
