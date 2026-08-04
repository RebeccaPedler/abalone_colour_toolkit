# Lip Segmentation Validation

## Method

n = 62 matched image pairs compared between manual lip segmentations and segmentations
produced by the Python code (`02_segment_ROI.py`).

- Manual segmentations were produced in Photoshop and exported as RGBA PNGs with a
  transparent background
- Python segmentations were exported as RGB PNGs with a white background
- Images were matched by filename, with the Python output identified by the `_ROI` suffix
- Overlap was quantified using Intersection over Union (IoU) and the Dice coefficient,
  calculated from binary masks derived from each image

IoU and Dice were calculated as:

> IoU = intersection / union
>
> Dice = 2 × intersection / (pixels_manual + pixels_python)

## Results

### Overall

| Metric | IoU | Dice |
|---|---|---|
| n (pairs) | 62 | 62 |
| Mean | 0.818 | 0.897 |
| Median | 0.839 | 0.912 |
| SD | 0.092 | 0.066 |
| Min | 0.362 | 0.532 |
| Max | 0.921 | 0.959 |

The large majority of images (n = 61, 98%) achieved IoU ≥ 0.5, and 75.8% achieved IoU ≥ 0.8.
One image (IMG_2617) was an outlier, with IoU = 0.362 and a Python/manual pixel ratio of
0.43, indicating the Python pipeline captured only ~43% of the manually segmented area.

### Distribution

| Percentile | IoU | Dice |
|---|---|---|
| 0% (min) | 0.362 | 0.532 |
| 25% | 0.806 | 0.893 |
| 50% (median) | 0.839 | 0.912 |
| 75% | 0.870 | 0.931 |
| 90% | 0.890 | 0.942 |
| 95% | 0.898 | 0.946 |
| 100% (max) | 0.921 | 0.959 |

### Threshold accuracy

| IoU threshold | Images meeting threshold (%) |
|---|---|
| ≥ 0.50 | 98.4% |
| ≥ 0.60 | 96.8% |
| ≥ 0.70 | 91.9% |
| ≥ 0.75 | 85.5% |
| ≥ 0.80 | 75.8% |
| ≥ 0.90 | 4.8% |

### Mask size comparison

Across the 57 images with IoU ≥ 0.7, the Python and manual masks were nearly the same size
(mean pixel ratio 0.95–0.97). In the small low-IoU group (IoU < 0.7, n = 5), the Python mask
was consistently smaller than the manual mask (mean ratio 0.75 for IoU 0.5–0.7; 0.43 for the
single outlier), indicating under-segmentation. We deem under-segmentation lower risk than a 
segmentation model which overestimates lip and captures other body components (e.g., foot).

| IoU band | n | Mean Python/manual pixel ratio |
|---|---|---|
| < 0.50 | 1 | 0.43 |
| 0.50–0.70 | 4 | 0.75 |
| 0.70–0.85 | 30 | 0.95 |
| 0.85–1.00 | 27 | 0.97 |

## Files

- `iou_results.csv` -- per-image IoU, Dice, and pixel counts for all 62 pairs
- `calculate_segmentation_performance.py` -- Python script used to compute masks and metrics
