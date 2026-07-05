#!/usr/bin/env python3
"""
segment_lips.py
---------------
Run your trained lip model over a folder of abalone photos (searching all
subfolders), and for each image save:
  * <name>_lip.png      - the lip on a white background (your cutout style),
  * <name>_overlay.jpg  - the original with the detected lip tinted, for QC,
and a summary.csv listing every image, whether a lip was found, and its size.

Handles CR3 raw files (needs `pip install rawpy`) and JPEG/PNG.

Pilot on 200 images spread across your subfolders:
    python segment_lips.py ^
        --weights "C:\\Users\\RebeccaPedler\\Documents\\lip_model\\runs\\lip_seg\\weights\\best.pt" ^
        --source  "D:\\AbalonePhotos" ^
        --out     "C:\\Users\\RebeccaPedler\\Documents\\lip_cutouts_test" ^
        --limit 200

Drop --limit to process everything.
"""

import argparse
import csv
import os
import random
from pathlib import Path
import numpy as np
import cv2

IMG_EXT = {".jpg", ".jpeg", ".png"}
RAW_EXT = {".cr3", ".cr2", ".nef", ".arw", ".dng"}
SKIP_SUFFIX = ("_lip", "_overlay", "copy")     # don't re-process our own outputs


def load_image(path: Path):
    """Return a BGR uint8 image, reading raw files via rawpy if needed."""
    if path.suffix.lower() in RAW_EXT:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.imread(str(path))


def find_images(root: Path):
    found = []
    for p in root.rglob("*"):
        if (p.is_file() and p.suffix.lower() in (IMG_EXT | RAW_EXT)
                and not p.stem.lower().endswith(SKIP_SUFFIX)):
            found.append(p)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--source", required=True, help="parent folder of photos")
    ap.add_argument("--out", required=True, help="where to write cutouts")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only this many images, spread across subfolders")
    ap.add_argument("--conf", type=float, default=0.3, help="detection confidence")
    ap.add_argument("--imgsz", type=int, default=1024, help="match training imgsz")
    ap.add_argument("--erode", type=int, default=3,
                    help="shrink the mask edge inward by N px to avoid edge bleed")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=["jpg", "png"], default="jpg",
                    help="cutout file type; jpg for quick tests, png (lossless) "
                         "for the final colour run")
    ap.add_argument("--flat", action="store_true",
                    help="put all cutouts directly in --out instead of mirroring "
                         "the source subfolders")
    ap.add_argument("--no-overlays", action="store_true",
                    help="skip the QC overlay images")
    args = ap.parse_args()

    source = Path(args.source).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Could not create output folder '{out}'.\n  {e}\n"
              "Use a local path outside OneDrive, e.g. "
              "C:\\Users\\RebeccaPedler\\Documents\\processed")
        return
    os.chdir(out)   # keep any writes on a known-writable, non-OneDrive path

    images = find_images(source)
    if not images:
        print(f"No images found under {source}")
        return
    random.Random(args.seed).shuffle(images)        # spread the sample around
    if args.limit:
        images = images[:args.limit]
    print(f"Processing {len(images)} image(s)...\n")

    from ultralytics import YOLO
    model = YOLO(args.weights)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rows, found, missed = [], 0, 0

    for i, path in enumerate(images, 1):
        bgr = load_image(path)
        if bgr is None:
            rows.append([str(path), "unreadable", 0]); missed += 1; continue
        h, w = bgr.shape[:2]

        res = model.predict(bgr, imgsz=args.imgsz, conf=args.conf,
                            retina_masks=True, device=args.device, verbose=False)[0]

        if res.masks is None or len(res.masks) == 0:
            rows.append([str(path), "no_lip_found", 0]); missed += 1
            print(f"[{i}/{len(images)}] {path.name}: no lip found")
            continue

        mask = (res.masks.data.cpu().numpy().max(0) > 0.5).astype(np.uint8) * 255
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if args.erode > 0:
            mask = cv2.erode(mask, kernel, iterations=args.erode)

        rel = path.relative_to(source)
        seg_dir = (out / "segmented") if args.flat else (out / "segmented" / rel.parent)
        seg_dir.mkdir(parents=True, exist_ok=True)

        copy = np.full_like(bgr, 255)
        copy[mask == 255] = bgr[mask == 255]
        ext = args.format
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == "jpg" else []
        cv2.imwrite(str(seg_dir / f"{path.stem}_lip.{ext}"), copy, params)

        if not args.no_overlays:
            poly_dir = (out / "polygons") if args.flat else (out / "polygons" / rel.parent)
            poly_dir.mkdir(parents=True, exist_ok=True)
            ov = bgr.copy()
            ov[mask == 255] = (0.4 * ov[mask == 255]
                               + np.array([0, 180, 0]) * 0.6).astype(np.uint8)
            cv2.imwrite(str(poly_dir / f"{path.stem}_overlay.jpg"), ov)

        frac = 100 * (mask > 0).mean()
        rows.append([str(path), "ok", round(frac, 3)]); found += 1
        print(f"[{i}/{len(images)}] {path.name}: lip = {frac:.2f}% of frame")

    with open(out / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows([["image", "status", "lip_pct_of_frame"]] + rows)

    print(f"\nFound a lip in {found} image(s); {missed} need a look.")
    print(f"Lip cutouts are under: {out / 'segmented'}")
    print(f"Annotated overlays are under: {out / 'polygons'}")
    print(f"summary.csv is in: {out}")
    print("Skim the overlays in 'polygons' and sort summary.csv by lip_pct to spot "
          "any misses (0%) or over-grabs (unusually large).")


if __name__ == "__main__":
    main()
