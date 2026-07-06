# Morphometric Validation

## Method

n = 2,261 paired measurements compared against `05_abalone_morphometrics.py` script
measurements on the same images.

- Manual measurements taken using a measuring board with ruler to the nearest whole millimetre
- Script version validated: `05_abalone_morphometrics.py`
- Comparison methods: Bland-Altman analysis, paired t-test, linear regression, and
  per-image absolute error with threshold breakdown
- Length was the only dimension validated; width and area have no manual reference measurements
- Three images were flagged by the script as outside the plausible size range
  (< 60 mm or > 130 mm) and analysed both included and excluded

## Results

### All data (n = 2,261)

| Metric | Value |
|---|---|
| MAE | 4.57 mm |
| RMSE | 5.68 mm |
| Mean bias (script − manual) | +3.93 mm |
| SD of differences | 4.10 mm |
| 95% Limits of Agreement | −4.10 to +11.96 mm |
| R² | 0.748 |
| Regression slope | 1.01 |
| Regression intercept | 2.98 mm |

### Flagged rows excluded (n = 2,258)

| Metric | Value |
|---|---|
| MAE | 4.50 mm |
| RMSE | 5.27 mm |
| Mean bias (script − manual) | +3.96 mm |
| SD of differences | 3.48 mm |
| 95% Limits of Agreement | −2.86 to +10.77 mm |

### Threshold accuracy (all data)

| Error threshold | Images within (%) |
|---|---|
| ≤ 1 mm | 7.7% |
| ≤ 2 mm | 18.0% |
| ≤ 3 mm | 31.1% |
| ≤ 5 mm | 60.6% |
| ≤ 10 mm | 97.6% |
| ≤ 15 mm | 99.2% |

### Absolute error distribution (all data)

| Percentile | Absolute error (mm) |
|---|---|
| 0% (min) | 0.00 |
| 25% | 2.55 |
| 50% (median) | 4.38 |
| 75% | 6.02 |
| 90% | 7.77 |
| 95% | 8.86 |
| 100% (max) | 66.26 |

### Key findings

The script systematically overestimates length by approximately 4 mm across the dataset.
Removing the three flagged gross-failure images reduced RMSE and tightened the upper limit of
agreement slightly but had no meaningful effect on bias (3.93 vs 3.96 mm).

Three distinct failure modes were identified from visual inspection of worst-performing images:

1. **Pale-shell segmentation failure** -- the HSV threshold (V < 110) excludes bright nacre
   but also excludes pale shell rims, causing the script to detect only the foot muscle (e.g., IMG_5762, error = −66 mm).
2. **Card-region contamination** -- when the abalone search box overlaps the ColorChecker card,
   the script measures card patches rather than abalone tissue, as ColorChecker colours satisfy
   the same HSV threshold (e.g., IMG_6357, error = +63 mm).
3. **Adequate segmentation** -- the majority of the dataset falls into this category, with a
   consistent ~4 mm positive bias likely attributable to the convex hull boundary capturing
   wet mantle tissue or shell margin.

Caveats on the reference method: Manual measurements are recorded to the nearest whole
millimetre, which places a minimum floor on agreement metrics and produces the vertical banding
visible in the Bland-Altman plot. There is also a possibility that manual measurements
systematically underreport true shell length. 

## Files

- `abalone_measurements.csv` -- paired measurements per image
- `bland_altman_comparison.png` -- Bland-Altman plots for all data and flagged rows excluded
- `morphometric_validation_performance.R` -- R script for reproducing all metrics and plots
