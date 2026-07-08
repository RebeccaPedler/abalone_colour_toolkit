# Colour Calibration Validation

## Method

n = 1,371 images processed through the colour calibration performed using the Python script `01_colour_correction_factors.py`.

- Each image was tested across four rotations to detect the ColorChecker card and where detected, eight possible grid orientations were
  tested, with the orientation giving the highest mean R² across all three channels selected as the best fit
- Patches with a detected L* below 8 or a reference L* below 20 were excluded from the fit
- A minimum of 10 valid patches was required for calibration
- For each of the L*, a*, and b* channels, a per-channel linear correction (slope and intercept) was fitted by least squares regression,  
  and fit was assessed by R²
- Calibration performance was measured using Delta E (dE), calculated before and after correction was applied
- Each calibrated image was assigned a quality score based on the R² and dE_after:
  - **Good**: mean R² > 0.85 and dE_after < 6
  - **Acceptable**: mean R² > 0.70 and dE_after < 10
  - **Poor**: does not meet either of the above
- Images where no checker could be detected in any rotation were labelled `no_checker_found` and excluded from calibration
- Images where a checker was detected but the correction fit failed were flagged `fit_failed`

Delta E before and after calibration was calculated as:

> dE_before = mean distance(observed, reference) across valid patches, prior to correction
> dE_after = mean distance(corrected, reference) across valid patches, following correction

## Results

### Overall

| Metric | dE before | dE after |
|---|---|---|
| n (calibrated images) | 1,339 | 1,339 |
| Mean | 31.154 | 6.985 |
| SD | 4.980 | 4.868 |
| Min | 11.759 | 4.455 |
| Max | 47.677 | 24.804 |
| Range | 35.918 | 20.349 |

Calibration reduced mean Delta E from 31.15 to 6.98 but 165 images (12.3% of calibrated images) still returned a dE > 10 after
calibration

### Quality categories

| Quality | n | % of calibrated images |
|---|---|---|
| Good | 1,130 | 84.39% |
| Acceptable | 44 | 3.29% |
| Poor | 165 | 12.32% |

### Channel fit (R²)

| Channel | Mean R² |
|---|---|
| L* | 0.935 |
| a* | 0.976 |
| b* | 0.972 |

L* showed the lowest mean fit of the three channels.

### Checker detection

| Outcome | n | % of total images |
|---|---|---|
| Calibrated | 1,339 | 97.66% |
| No checker found | 31 | 2.26% |
| Fit failed | 1 | 0.07% |
| **Total** | **1,371** | **100%** |

Mean number of checker patches detected across calibrated images was 20.29 (out of a possible 24).

## Findings

The pipeline performed well overall, with the large majority of images (84.4%) achieving a
good calibration and a further 3.3% achieving an acceptable calibration. 12.3% of
calibrated images (n = 165) were classed as poor. It is up to the discretion of the users of this
pipeline as to whether to exclude images on the basis of dE.

Mean R² was highest for a* and b* (0.976 and 0.972) and lowest for L* (0.935).

A small number of images (2.26%) had no checker detected at all and were excluded from calibration.

## Files

- `correction_factors.csv` -- per-image calibration diagnostics, including quality
  category, per-channel slope/intercept/R², patch count, and dE before/after for all
  1,371 images
- `calibration_summary.csv` -- summary statistics generated from `correction_factors.csv`
- `colour_correction_performance.R` -- R script used to compute the summary statistics
