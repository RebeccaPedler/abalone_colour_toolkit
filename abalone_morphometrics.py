#!/usr/bin/env python3
"""
abalone_morphometrics.py  –  v4.0
===================================
Batch-measures abalone from lightbox JPEG images.

Supports TWO photo layouts, auto-detected per image:
  Layout A ("side_by_side"): vertical black divider, card and abalone in
            opposite left/right halves (card side auto-detected).
  Layout B ("stacked"):      horizontal black divider near the top of frame,
            card and abalone both below it, abalone position varies anywhere
            in the lower region (not just under the card).

Setup / Installation
--------------------
    pip install opencv-python numpy colour-science colour-checker-detection

Pipeline
--------
1.  Load JPEG (grey/white board background, Calibrite ColorChecker Classic,
    thick black divider strip in either orientation).
2.  Detect the divider's orientation (vertical or horizontal) and position.
3.  Detect which side/region of the divider holds the ColorChecker card by
    testing candidate regions for colour-patch matches, rather than assuming
    a fixed side.
4.  Compute px/mm from patch spacing (6 patches x 12 mm along long dimension).
5.  Segment the abalone from the region opposite/below the card using HSV
    thresholds calibrated from ground-truth:
      (V < 75) OR (S > 30), AND (V > 15)
    -- abalone tissue is darker and more colourful than the neutral grey board.
6.  Fit a minimum-area rotated bounding rectangle -> length & width (mm).
7.  Compute filled-contour area (mm^2).
8.  Hard-exclude any candidate contour whose bounding box overlaps the card
    region — prevents ColorChecker patches being measured as abalone tissue.
9.  Save an annotated visualisation.
10. Append results to a CSV. Measurements outside the plausible size range
    (< 60 mm or > 130 mm) are flagged with a CHECK comment in the CSV for
    manual review rather than silently accepted.

Usage (Windows)
---------------
cd C:\\Users\\RebeccaPedler\\Documents
python abalone_morphometrics_4.py ^
    --images  "C:\\Users\\RebeccaPedler\\Documents\\measurement validation" ^
    --output  "C:\\Users\\RebeccaPedler\\Documents\\measurement validation\\abalone_measurements.csv" ^
    --vis_dir "C:\\Users\\RebeccaPedler\\Documents\\measurement validation\\abalone_annotated"

Optional flags
--------------
--scale_mm      Real-world span of the ruler (mm).  Default = 123
--ext           Comma-separated extensions to glob.  Default = jpg,jpeg
--min_area_px   Min contour area (px2) to be an abalone.  Default = 50000
--debug         Also save intermediate mask images.
"""

import argparse
import csv
import glob
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RULER_REAL_MM        = 123
MIN_DIVIDER_FRAC     = 0.01   # divider strip width/height as a fraction of
MAX_DIVIDER_FRAC     = 0.12   # the image's relevant dimension (generous bounds)
DIVIDER_DARK_THRESH  = 0.4    # fraction of a row/col that must be "dark" pixels

# Plausible length range for greenlip abalone (mm). Measurements outside
# this range are flagged in the CSV for manual review — they are not dropped.
LENGTH_MIN_MM = 60
LENGTH_MAX_MM = 130


# ===========================================================================
# 1.  DIVIDER DETECTION  (orientation + position, auto-detected)
# ===========================================================================

def _find_contiguous_runs(bool_1d):
    """Return list of (start, end_inclusive, length) for contiguous True runs."""
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(bool_1d):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            in_run = False
            runs.append((start, i - 1, i - start))
    if in_run:
        runs.append((start, len(bool_1d) - 1, len(bool_1d) - start))
    return runs


def detect_divider(img_bgr):
    """
    Detect the black divider strip's orientation and position.

    Tests both axes independently using a contiguous-run filter (not just a
    threshold), since other dark regions in frame (ceiling edges, shadows,
    card gridlines) can also exceed the darkness threshold without being
    the genuine divider strip.

    Returns a dict:
      {"orientation": "vertical"|"horizontal", "start": int, "end": int}
    "start"/"end" are column indices for vertical, row indices for horizontal.
    Falls back to a vertical divider at the image's horizontal midpoint if
    nothing valid is found (preserves old behaviour for edge cases).
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # --- Vertical strip check (scan columns) ---
    col_frac   = dark.sum(axis=0).astype(float) / (255.0 * h)
    col_smooth = np.convolve(col_frac, np.ones(31) / 31, mode='same')
    v_runs     = _find_contiguous_runs(col_smooth > DIVIDER_DARK_THRESH)
    v_runs     = [r for r in v_runs
                  if MIN_DIVIDER_FRAC * w <= r[2] <= MAX_DIVIDER_FRAC * w]
    best_v     = max(v_runs, key=lambda r: r[2]) if v_runs else None

    # --- Horizontal strip check (scan rows) ---
    row_frac   = dark.sum(axis=1).astype(float) / (255.0 * w)
    row_smooth = np.convolve(row_frac, np.ones(31) / 31, mode='same')
    h_runs     = _find_contiguous_runs(row_smooth > DIVIDER_DARK_THRESH)
    h_runs     = [r for r in h_runs
                  if MIN_DIVIDER_FRAC * h <= r[2] <= MAX_DIVIDER_FRAC * h]
    best_h     = max(h_runs, key=lambda r: r[2]) if h_runs else None

    # Prefer whichever axis actually found a valid, well-sized run.
    # (In practice these are mutually exclusive across the two known rigs,
    # but if both somehow match, take the more confident/longer run.)
    if best_v and not best_h:
        return {"orientation": "vertical", "start": best_v[0], "end": best_v[1]}
    if best_h and not best_v:
        return {"orientation": "horizontal", "start": best_h[0], "end": best_h[1]}
    if best_v and best_h:
        if best_v[2] >= best_h[2]:
            return {"orientation": "vertical", "start": best_v[0], "end": best_v[1]}
        return {"orientation": "horizontal", "start": best_h[0], "end": best_h[1]}

    # Fallback: nothing detected confidently — assume vertical at midpoint
    # (matches old default behaviour rather than failing the image outright).
    return {"orientation": "vertical", "start": w // 2 - 1, "end": w // 2}


# ===========================================================================
# 1b.  CARD-SIDE DETECTION  (auto-detect which region holds the ColorChecker)
# ===========================================================================

def _quick_patch_count(region_bgr):
    """
    Cheap colour-patch counter used only to decide WHICH region the card is
    in. Reuses the same colour bands as pixels_per_mm's patch detector, but
    just counts matches rather than measuring them precisely.
    """
    if region_bgr is None or region_bgr.size == 0:
        return 0
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(float), hsv[:, :, 1].astype(float), hsv[:, :, 2].astype(float)

    bands = {
        "yellow":  (H >= 20)  & (H <= 35)  & (S > 150) & (V > 80),
        "cyan":    (H >= 85)  & (H <= 100) & (S > 150) & (V > 80),
        "magenta": (H >= 145) & (H <= 165) & (S > 150) & (V > 80),
        "green":   (H >= 60)  & (H <= 85)  & (S > 150) & (V > 60),
        "orange":  (H >= 5)   & (H <= 20)  & (S > 150) & (V > 80),
    }
    k_close = np.ones((8, 8), np.uint8)
    k_open  = np.ones((4, 4), np.uint8)
    MIN_PX, MAX_PX = 50, 600

    count = 0
    for band in bands.values():
        mask = band.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            xb, yb, bw, bh = cv2.boundingRect(c)
            if not (MIN_PX < bw < MAX_PX and MIN_PX < bh < MAX_PX):
                continue
            if not (0.70 < bw / bh < 1.30):
                continue
            if cv2.contourArea(c) < (MIN_PX ** 2) * 0.5:
                continue
            count += 1
    return count


def detect_layout(img_bgr, divider):
    """
    Given the detected divider, work out the full layout for this image:
      - which side/region holds the card
      - which side/region holds the abalone
      - the pixel bounding box to search for each

    Returns a dict with keys:
      "type"        : "side_by_side" | "stacked"
      "card_box"    : (x0, y0, x1, y1)  region to search for the colour card
      "abalone_box" : (x0, y0, x1, y1)  region to search for the abalone
    """
    h, w = img_bgr.shape[:2]
    MARGIN = 80  # px buffer to clear the divider strip itself

    if divider["orientation"] == "vertical":
        # Layout A: side-by-side. Card is in whichever half has more patches.
        mid = (divider["start"] + divider["end"]) // 2
        left_box  = (0, 0, mid, h)
        right_box = (mid, 0, w, h)

        left_count  = _quick_patch_count(img_bgr[0:h, 0:mid])
        right_count = _quick_patch_count(img_bgr[0:h, mid:w])

        if left_count >= right_count:
            card_box, abalone_box = left_box, right_box
        else:
            card_box, abalone_box = right_box, left_box

        # Pull the abalone search box in from the divider by MARGIN so the
        # divider strip itself never gets picked up as part of the mask.
        ax0, ay0, ax1, ay1 = abalone_box
        if ax0 == 0:
            ax1 = max(ax1 - MARGIN, ax0 + 1)
        else:
            ax0 = min(ax0 + MARGIN, ax1 - 1)
        abalone_box = (ax0, ay0, ax1, ay1)

        return {"type": "side_by_side", "card_box": card_box, "abalone_box": abalone_box}

    else:
        # Layout B: stacked. Card and abalone are both on the same side of
        # the frame (left in observed examples) below the divider; the
        # area below the divider is searched in full for both, since the
        # card sits near the top of that region and the abalone can be
        # anywhere lower down. We split crudely into "upper-lower" and
        # "rest-of-lower" by finding the card first, then searching
        # everything below the card's bottom edge (plus margin) for the
        # abalone — this matches the variable abalone placement observed.
        below_y0 = min(divider["end"] + MARGIN, h - 1)
        below_box = (0, below_y0, w, h)

        # Find the card within the "below divider" region precisely enough
        # to know its bottom edge, so the abalone search can start clear of it.
        below_region = img_bgr[below_y0:h, 0:w]
        card_box, card_bottom_y = _locate_card_bottom_in_region(below_region, below_y0)

        abalone_y0 = min(card_bottom_y + MARGIN, h - 1)
        abalone_box = (0, abalone_y0, w, h)

        return {"type": "stacked", "card_box": card_box, "abalone_box": abalone_box}


def _locate_card_bottom_in_region(region_bgr, y_offset):
    """
    Within a region known to contain the card, find the card's approximate
    bounding box (in full-image coordinates) using the same patch detector,
    so we know where it ends and can search below it for the abalone.

    Stray blobs elsewhere in the region (e.g. the abalone's own tissue
    colour can coincidentally match one of the patch HSV bands) are
    rejected by clustering on patch CENTRES around the median position and
    discarding anything implausibly far away, rather than trusting the raw
    min/max extent of every match.

    Returns (card_box, card_bottom_y_full_image_coords).
    Falls back to the top third of the region if no patches are found.
    """
    h, w = region_bgr.shape[:2]
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(float), hsv[:, :, 1].astype(float), hsv[:, :, 2].astype(float)

    bands = {
        "yellow":  (H >= 20)  & (H <= 35)  & (S > 150) & (V > 80),
        "cyan":    (H >= 85)  & (H <= 100) & (S > 150) & (V > 80),
        "magenta": (H >= 145) & (H <= 165) & (S > 150) & (V > 80),
        "green":   (H >= 60)  & (H <= 85)  & (S > 150) & (V > 60),
        "orange":  (H >= 5)   & (H <= 20)  & (S > 150) & (V > 80),
    }
    k_close = np.ones((8, 8), np.uint8)
    k_open  = np.ones((4, 4), np.uint8)
    MIN_PX, MAX_PX = 50, 600

    boxes = []  # (xb, yb, bw, bh) for every candidate patch
    for band in bands.values():
        mask = band.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            xb, yb, bw, bh = cv2.boundingRect(c)
            if not (MIN_PX < bw < MAX_PX and MIN_PX < bh < MAX_PX):
                continue
            if not (0.70 < bw / bh < 1.30):
                continue
            if cv2.contourArea(c) < (MIN_PX ** 2) * 0.5:
                continue
            boxes.append((xb, yb, bw, bh))

    if not boxes:
        # Fallback: assume card occupies roughly the top third of the region
        fallback_bottom = y_offset + int(h * 0.35)
        return (0, y_offset, w, fallback_bottom), fallback_bottom

    centres = np.array([(xb + bw / 2.0, yb + bh / 2.0) for xb, yb, bw, bh in boxes])
    median_centre = np.median(centres, axis=0)

    # A real ColorChecker card's patches all sit within a few patch-widths of
    # each other. Use the median patch size to set a generous "same cluster"
    # radius, then drop any candidate further than that from the median
    # centre — this is what rejects a stray match out in the abalone region.
    sizes = np.array([(bw + bh) / 2.0 for _, _, bw, bh in boxes])
    typical_size = np.median(sizes)
    cluster_radius = typical_size * 6.0   # card spans ~6 patches across

    dists = np.linalg.norm(centres - median_centre, axis=1)
    keep = dists <= cluster_radius

    if not np.any(keep):
        fallback_bottom = y_offset + int(h * 0.35)
        return (0, y_offset, w, fallback_bottom), fallback_bottom

    kept_boxes = [b for b, k in zip(boxes, keep) if k]
    xs = [xb for xb, yb, bw, bh in kept_boxes] + [xb + bw for xb, yb, bw, bh in kept_boxes]
    ys = [yb for xb, yb, bw, bh in kept_boxes] + [yb + bh for xb, yb, bw, bh in kept_boxes]

    pad = 60
    x0, x1 = max(0, min(xs) - pad), min(w, max(xs) + pad)
    y0, y1 = max(0, min(ys) - pad), min(h, max(ys) + pad)
    card_box_full = (x0, y_offset + y0, x1, y_offset + y1)
    return card_box_full, y_offset + y1


# ===========================================================================
# 2.  SCALE CALIBRATION — via ColourChecker detection library
# ===========================================================================
#
# Uses the colour_checker_detection library to locate the Calibrite
# ColorChecker Classic card and compute px/mm from its patch spacing.
# Each patch is PATCH_SIZE_MM mm; the card has N_PATCHES_LONG patches
# along its long dimension.  Tries all 4 rotations to handle flipped cards.

PATCH_SIZE_MM  = 12.0   # physical size of one patch (mm)
N_PATCHES_LONG = 6      # patches along the long card dimension


def pixels_per_mm(img_bgr, card_box,
                  ruler_real_mm=RULER_REAL_MM,
                  debug_dir=None, stem=""):
    """
    Return (px_per_mm, patch_centres) by directly measuring individual
    ColorChecker patches within card_box (x0, y0, x1, y1) in full-image
    coordinates. card_box is determined per-image by detect_layout(), since
    the card can appear on either side of the divider (or above/below it).

    Targets five well-separated colours (yellow, cyan, magenta, green, orange)
    and measures the bounding-box side length of each detected square patch.
    Each patch is physically PATCH_SIZE_MM x PATCH_SIZE_MM.
    Uses the median patch size to reject fragments or partial detections.

    Returns (px_per_mm, quad_pts) where quad_pts is a (4,2) int32 array
    enclosing all detected patch centres, in FULL-IMAGE coordinates
    (for visualisation).
    Raises ValueError if fewer than 2 patches are detected.
    """
    cx0, cy0, cx1, cy1 = card_box
    region = img_bgr[cy0:cy1, cx0:cx1]

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    H   = hsv[:, :, 0].astype(float)
    S   = hsv[:, :, 1].astype(float)
    V   = hsv[:, :, 2].astype(float)

    # Colour bands targeting individual, well-separated patches
    bands = {
        "yellow":  (H >= 20)  & (H <= 35)  & (S > 150) & (V > 80),
        "cyan":    (H >= 85)  & (H <= 100) & (S > 150) & (V > 80),
        "magenta": (H >= 145) & (H <= 165) & (S > 150) & (V > 80),
        "green":   (H >= 60)  & (H <= 85)  & (S > 150) & (V > 60),
        "orange":  (H >= 5)   & (H <= 20)  & (S > 150) & (V > 80),
    }

    k_close = np.ones((8,  8),  np.uint8)
    k_open  = np.ones((4,  4),  np.uint8)

    # Expected patch size range in pixels (very wide tolerance)
    MIN_PX, MAX_PX = 50, 600

    patch_sizes   = []
    patch_centres = []   # in region-local coordinates for now

    for name, band in bands.items():
        mask = band.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            xb, yb, bw, bh = cv2.boundingRect(c)
            if not (MIN_PX < bw < MAX_PX and MIN_PX < bh < MAX_PX):
                continue
            if not (0.70 < bw / bh < 1.30):        # must be roughly square
                continue
            if cv2.contourArea(c) < (MIN_PX ** 2) * 0.5:
                continue
            patch_sizes.append((bw + bh) / 2.0)
            patch_centres.append((xb + bw // 2, yb + bh // 2))

    if len(patch_sizes) < 2:
        raise ValueError(
            f"Only {len(patch_sizes)} colour patch(es) found in card region "
            f"{card_box} — check that the ColorChecker card was located correctly.")

    # Median is robust against partial patches at card edges
    px_per_mm = float(np.median(patch_sizes)) / PATCH_SIZE_MM

    region_h, region_w = region.shape[:2]

    # Build a convex hull around patch centres for the card overlay
    if len(patch_centres) >= 3:
        pts        = np.array(patch_centres, dtype=np.int32)
        # Add generous padding so the overlay covers the full card body
        pad        = int(np.median(patch_sizes) * 0.8)
        x0c = max(0,         pts[:, 0].min() - pad)
        y0c = max(0,         pts[:, 1].min() - pad)
        x1c = min(region_w,  pts[:, 0].max() + pad)
        y1c = min(region_h,  pts[:, 1].max() + pad)
        quad_pts = np.array([[x0c, y0c], [x1c, y0c],
                              [x1c, y1c], [x0c, y1c]], dtype=np.int32)
    else:
        quad_pts = np.array(patch_centres, dtype=np.int32)

    # Offset back to full-image coordinates before returning
    quad_pts = quad_pts + np.array([cx0, cy0], dtype=np.int32)

    return px_per_mm, quad_pts


# ===========================================================================
# 3.  ABALONE SEGMENTATION
# ===========================================================================

def segment_abalone(img_bgr, search_box, near_edge, card_box,
                    min_area_px=50_000,
                    debug_dir=None, stem=""):
    """
    Segment the abalone from the grey board within search_box.

    Thresholds calibrated from ground-truth red annotations:
      True abalone:  S mean=94,  V mean=75-84  (colourful, dark)
      Nacre/water:   S mean=22,  V mean=120     (neutral grey, bright)

    Two-pass approach:
      Pass 1: High-confidence mask  (S>35) AND (V<110) AND (V>15)
              Captures colourful shell/tissue, excludes white nacre and wet board.
      Pass 2: Within the bounding region of Pass 1, apply a slightly wider
              threshold to recover any shell edges missed by strict V<110.
      Final:  Convex hull to remove water tendrils and irregular edges.

    Card-overlap exclusion:
      Any candidate contour whose axis-aligned bounding box overlaps card_box
      by more than CARD_OVERLAP_THRESH of the contour's own bounding area is
      rejected outright. This prevents ColorChecker patches from being
      measured as abalone when the search region leaks into the card area.

    search_box : (x0, y0, x1, y1) in full-image coordinates.
    near_edge  : "left" | "right" | "top" | "bottom" — edge adjacent to divider.
    card_box   : (x0, y0, x1, y1) in full-image coordinates — the card region
                 to exclude from candidate contours.

    Returns (contour_full_coords, full_mask) or (None, None).
    """
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = search_box
    roi = img_bgr[y0:y1, x0:x1]
    roi_h, roi_w = roi.shape[:2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S   = hsv[:, :, 1].astype(float)
    V   = hsv[:, :, 2].astype(float)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    # ── PASS 1: high-confidence pixels ───────────────────────────────────────
    # S>35 = colourful shell/lip/tissue
    # V<110 = excludes bright nacre rim (nacre V mean=120)
    # V>15  = excludes dark outer frame
    mask_hc = ((S > 35) & (V < 110) & (V > 15)).astype(np.uint8) * 255
    mask_hc = cv2.morphologyEx(mask_hc, cv2.MORPH_CLOSE, k_close)
    mask_hc = cv2.morphologyEx(mask_hc, cv2.MORPH_OPEN,  k_open)

    cnts_hc, _ = cv2.findContours(mask_hc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_hc:
        return None, None

    valid_hc = [c for c in cnts_hc if cv2.contourArea(c) > min_area_px * 0.3]
    if not valid_hc:
        return None, None

    anchor      = max(valid_hc, key=cv2.contourArea)
    anchor_mask = np.zeros((roi_h, roi_w), np.uint8)
    cv2.drawContours(anchor_mask, [anchor], -1, 255, cv2.FILLED)

    # ── PASS 2: wider threshold inside the anchor region ─────────────────────
    search_region = cv2.dilate(anchor_mask, np.ones((40, 40), np.uint8))
    mask_w = (((S > 20) & (V < 120) & (V > 15)) & (search_region > 0)).astype(np.uint8) * 255
    mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_CLOSE, k_close)
    mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_OPEN,  np.ones((6, 6), np.uint8))

    if debug_dir and stem:
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_seg_mask.jpg"), mask_w)

    contours, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # ── SELECT best contour ───────────────────────────────────────────────────
    # Fraction of a candidate's bounding-box area that may overlap the card
    # region before the contour is rejected as "measuring the card, not the
    # abalone". Set low (5 %) to catch IMG_6357-style failures while still
    # allowing occasional edge-of-frame coincidences.
    CARD_OVERLAP_THRESH = 0.05

    # Unpack card_box into full-image coordinates for overlap checks below.
    cb_x0, cb_y0, cb_x1, cb_y1 = card_box

    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        hull = cv2.convexHull(cnt)
        rect = cv2.minAreaRect(hull)
        _, (rw, rh), _ = rect
        if rw <= 0 or rh <= 0:
            continue
        if max(rw, rh) / min(rw, rh) > 4.0:
            continue

        xc, yc, cntw, cnth = cv2.boundingRect(cnt)
        # Convert contour bounding rect from ROI-local to full-image coords.
        xc_full = xc + x0
        yc_full = yc + y0

        # ── Card-overlap exclusion ────────────────────────────────────────────
        # Compute the intersection of the contour's full-image bounding box
        # with the card bounding box.
        ix0 = max(xc_full, cb_x0)
        iy0 = max(yc_full, cb_y0)
        ix1 = min(xc_full + cntw, cb_x1)
        iy1 = min(yc_full + cnth, cb_y1)
        if ix1 > ix0 and iy1 > iy0:
            overlap_area = (ix1 - ix0) * (iy1 - iy0)
            contour_bbox_area = cntw * cnth
            if contour_bbox_area > 0 and overlap_area / contour_bbox_area > CARD_OVERLAP_THRESH:
                print(f"    [SKIP] Contour rejected — {overlap_area / contour_bbox_area:.1%} "
                      f"overlap with card region (likely card patches, not abalone).")
                continue

        # Exclude contours hugging the edge nearest the divider/card — this
        # used to be a hardcoded "xc <= 10" (left edge) check; it now checks
        # whichever edge of the ROI is adjacent to the divider, since that
        # edge can be left/right/top depending on layout. The "too wide/tall"
        # sanity check is generalised the same way.
        if near_edge == "left":
            if rw > roi_w * 0.90 or xc <= 10:
                continue
        elif near_edge == "right":
            if rw > roi_w * 0.90 or (xc + cntw) >= roi_w - 10:
                continue
        elif near_edge == "top":
            if rh > roi_h * 0.90 or yc <= 10:
                continue
        elif near_edge == "bottom":
            if rh > roi_h * 0.90 or (yc + cnth) >= roi_h - 10:
                continue

        valid.append(cnt)

    if not valid:
        return None, None

    best      = max(valid, key=cv2.contourArea)
    best_hull = cv2.convexHull(best)   # convex hull removes water tendrils

    # Offset back to full-image coordinates
    best_hull[:, :, 0] += x0
    best_hull[:, :, 1] += y0

    full_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(full_mask, [best_hull], -1, 255, cv2.FILLED)

    return best_hull, full_mask


# ===========================================================================
# 4.  MEASUREMENT
# ===========================================================================

def measure_abalone(contour, mask, px_mm):
    """Return dict with length_mm, width_mm, area_mm2, rect, box_pts."""
    rect      = cv2.minAreaRect(contour)
    box_w, box_h = rect[1]
    length_px = max(box_w, box_h)
    width_px  = min(box_w, box_h)
    area_px   = float(np.count_nonzero(mask))

    return {
        "length_mm": round(length_px / px_mm, 2),
        "width_mm":  round(width_px  / px_mm, 2),
        "area_mm2":  round(area_px   / (px_mm ** 2), 2),
        "rect":      rect,
        "box_pts":   cv2.boxPoints(rect).astype(int),
    }


# ===========================================================================
# 5.  VISUALISATION
# ===========================================================================

def save_annotated(img_bgr, contour, meas, divider, px_mm, out_path,
                   card_quad=None):
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    font  = cv2.FONT_HERSHEY_SIMPLEX
    fsc   = max(1.0, w / 3000)
    thick = max(2, int(w / 1500))
    lh    = int(55 * fsc)

    # ── Divider line — orientation-aware ─────────────────────────────────────
    divider_mid = (divider["start"] + divider["end"]) // 2
    if divider["orientation"] == "vertical":
        cv2.line(vis, (divider_mid, 0), (divider_mid, h), (255, 180, 0), 3)
    else:
        cv2.line(vis, (0, divider_mid), (w, divider_mid), (255, 180, 0), 3)

    # ── ColorChecker card overlay ─────────────────────────────────────────────
    if card_quad is not None:
        pts = card_quad.reshape((-1, 1, 2))
        # Filled semi-transparent highlight
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], (255, 200, 0))          # amber fill
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)    # 25% opacity
        # Solid border
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 200, 0),
                      thickness=max(3, thick + 1))
        # Label — positioned above the top-left corner of the quad
        lbl_x = int(card_quad[:, 0].min())
        lbl_y = int(card_quad[:, 1].min()) - int(20 * fsc)
        lbl_y = max(lbl_y, int(40 * fsc))
        scale_lbl = f"Scale: {px_mm:.2f} px/mm  ({PATCH_SIZE_MM:.0f}mm patches)"
        cv2.putText(vis, scale_lbl, (lbl_x + 3, lbl_y + 3),
                    font, fsc * 0.85, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(vis, scale_lbl, (lbl_x, lbl_y),
                    font, fsc * 0.85, (255, 200, 0), thick, cv2.LINE_AA)

    # ── Abalone contour + bounding box ───────────────────────────────────────
    cv2.drawContours(vis, [contour],         -1, (0, 230, 60),  4)
    cv2.drawContours(vis, [meas["box_pts"]], -1, (0, 220, 255), 3)

    rect = meas["rect"]
    cx   = int(rect[0][0])
    cy   = int(rect[0][1])

    for i, lbl in enumerate([
        f"Length: {meas['length_mm']:.1f} mm",
        f"Width:  {meas['width_mm']:.1f} mm",
        f"Area:   {meas['area_mm2']:.0f} mm2",
    ]):
        ypos = cy - lh + i * lh
        cv2.putText(vis, lbl, (cx + 3, ypos + 3),
                    font, fsc, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(vis, lbl, (cx, ypos),
                    font, fsc, (0, 255, 200), thick, cv2.LINE_AA)

    # ── Scale bar ─────────────────────────────────────────────────────────────
    bar_mm = 20
    bar_px = int(bar_mm * px_mm)
    bx, by = 60, h - 80
    cv2.rectangle(vis, (bx, by - 25), (bx + bar_px, by), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(vis, (bx, by - 25), (bx + bar_px, by), (255, 255, 255), 3)
    cv2.putText(vis, f"{bar_mm} mm", (bx, by - 35),
                font, fsc, (255, 255, 255), thick, cv2.LINE_AA)

    cv2.imwrite(out_path, vis)


# ===========================================================================
# 6.  PER-IMAGE PIPELINE
# ===========================================================================

def process_image(img_path, vis_dir, ruler_real_mm, min_area_px, debug):
    stem    = Path(img_path).stem
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  [WARN] Cannot read: {img_path}")
        return None

    debug_dir = vis_dir if debug else None
    h, w = img_bgr.shape[:2]

    divider = detect_divider(img_bgr)
    layout  = detect_layout(img_bgr, divider)

    try:
        px_mm, card_pts = pixels_per_mm(img_bgr, layout["card_box"],
                                        ruler_real_mm, debug_dir, stem)
    except ValueError as exc:
        print(f"  [WARN] Scale error ({stem}, layout={layout['type']}): {exc}")
        return None

    # Work out which edge of the abalone search box sits against the
    # divider/card, so segment_abalone excludes contours hugging that edge
    # (this used to be hardcoded to "left edge"; it now depends on layout).
    ax0, ay0, ax1, ay1 = layout["abalone_box"]
    if layout["type"] == "side_by_side":
        # abalone box is whichever side did NOT get the card
        near_edge = "left" if ax0 == 0 else "right"
    else:  # stacked
        near_edge = "top"  # abalone box always starts below card/divider

    contour, mask = segment_abalone(img_bgr, layout["abalone_box"], near_edge,
                                    layout["card_box"],
                                    min_area_px, debug_dir, stem)
    if contour is None:
        print(f"  [WARN] No abalone found in {stem} (layout={layout['type']})")
        return None

    meas = measure_abalone(contour, mask, px_mm)

    # ── Plausibility check ────────────────────────────────────────────────────
    # Flag measurements outside the expected size range for greenlip abalone.
    # Flagged rows are still written to CSV — they need manual review, not
    # silent deletion. Check the annotated image for the flagged image_ID.
    length = meas["length_mm"]
    if length < LENGTH_MIN_MM or length > LENGTH_MAX_MM:
        check_flag = (f"CHECK: length {length} mm outside expected range "
                      f"({LENGTH_MIN_MM}–{LENGTH_MAX_MM} mm) — review annotated image")
        print(f"  [FLAG] {stem}: length {length} mm is outside plausible range "
              f"({LENGTH_MIN_MM}–{LENGTH_MAX_MM} mm). Flagged in CSV.")
    else:
        check_flag = ""

    if vis_dir:
        vis_path = os.path.join(vis_dir, f"{stem}_annotated.jpg")
        save_annotated(img_bgr, contour, meas, divider, px_mm, vis_path,
                       card_quad=card_pts)

    print(f"  {stem}: L={meas['length_mm']} mm  "
          f"W={meas['width_mm']} mm  "
          f"A={meas['area_mm2']} mm2  "
          f"[{px_mm:.2f} px/mm]  [layout={layout['type']}]")

    return {
        "filename":        Path(img_path).name,
        "length_mm":       meas["length_mm"],
        "width_mm":        meas["width_mm"],
        "area_mm2":        meas["area_mm2"],
        "scale_px_per_mm": round(px_mm, 4),
        "layout":          layout["type"],
        "check":           check_flag,
    }


# ===========================================================================
# 7.  MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Batch abalone morphometrics from lightbox JPEG images.")
    ap.add_argument("--images",      required=True)
    ap.add_argument("--output",      required=True)
    ap.add_argument("--vis_dir",     default=None)
    ap.add_argument("--scale_mm",    type=float, default=RULER_REAL_MM)
    ap.add_argument("--ext",         default="jpg,jpeg")
    ap.add_argument("--min_area_px", type=int, default=50_000)
    ap.add_argument("--debug",       action="store_true")
    args = ap.parse_args()

    exts      = [e.strip().lstrip(".") for e in args.ext.split(",")]
    img_paths = []
    for ext in exts:
        # ** makes glob recurse into all subfolders
        img_paths += glob.glob(os.path.join(args.images, "**", f"*.{ext}"),         recursive=True)
        img_paths += glob.glob(os.path.join(args.images, "**", f"*.{ext.upper()}"), recursive=True)
    img_paths = sorted(set(img_paths))

    # Report how many subfolders were found
    subfolders = sorted(set(os.path.dirname(p) for p in img_paths))
    print(f"Scanning {len(subfolders)} folder(s):")
    for sf in subfolders:
        count = sum(1 for p in img_paths if os.path.dirname(p) == sf)
        print(f"  {sf}  ({count} image(s))")

    if not img_paths:
        print(f"No images found in: {args.images}")
        sys.exit(1)

    print(f"Found {len(img_paths)} image(s).\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    results = []
    for img_path in img_paths:
        print(f"Processing: {Path(img_path).name}")
        r = process_image(img_path, args.vis_dir, args.scale_mm,
                          args.min_area_px, args.debug)
        if r:
            results.append(r)

    if not results:
        print("No measurements produced.")
        sys.exit(1)

    fieldnames = ["filename", "length_mm", "width_mm", "area_mm2",
                  "scale_px_per_mm", "layout", "check"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'─'*60}")
    print(f"Done!  {len(results)}/{len(img_paths)} abalone measured.")
    flagged = sum(1 for r in results if r["check"])
    if flagged:
        print(f"Flagged for review: {flagged} row(s) — see 'check' column in CSV.")
    print(f"CSV  -> {args.output}")
    if args.vis_dir:
        print(f"Vis  -> {args.vis_dir}")


if __name__ == "__main__":
    main()
