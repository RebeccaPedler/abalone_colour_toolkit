# Colour Calibration Assessment

# Assesses calibration pipeline performance from correction_factors.csv

library(here)
library(dplyr)

# Load data
df <- read.csv("correction_factors.csv", stringsAsFactors = FALSE)

# Split into checker-found vs no-checker rows
found   <- df %>% filter(status == "calibrated")
no_chk  <- df %>% filter(status == "no_checker_found")

# Delta E before/after summary stats 
dE_stats <- function(x) {
  x <- x[!is.na(x)]
  c(mean = mean(x), sd = sd(x), min = min(x), max = max(x), range = max(x) - min(x))
}

before <- dE_stats(found$dE_before)
after  <- dE_stats(found$dE_after)

n_poor_after <- sum(found$dE_after > 10, na.rm = TRUE)

# Quality category counts (good / acceptable / poor) 
n_total <- nrow(found)
quality_counts <- found %>%
  count(quality) %>%
  mutate(pct = round(100 * n / n_total, 2))

get_q <- function(q, col) {
  val <- quality_counts[[col]][quality_counts$quality == q]
  if (length(val) == 0) 0 else val
}

# Mean R2 per CIELAB channel
r2_L <- mean(found$L_r2, na.rm = TRUE)
r2_a <- mean(found$a_r2, na.rm = TRUE)
r2_b <- mean(found$b_r2, na.rm = TRUE)

# No checker found stats 
n_no_checker <- nrow(no_chk)
pct_no_checker <- round(100 * n_no_checker / nrow(df), 2)

# Average n_patches (calibrated images only) 
avg_n_patches <- mean(found$n_patches, na.rm = TRUE)

# Assemble summary table
summary_df <- data.frame(
  metric = c(
    "n_images_total", "n_calibrated", "n_no_checker_found", "pct_no_checker_found",
    "avg_n_patches",
    "dE_before_mean", "dE_before_sd", "dE_before_min", "dE_before_max", "dE_before_range",
    "dE_after_mean", "dE_after_sd", "dE_after_min", "dE_after_max", "dE_after_range",
    "n_dE_after_gt10",
    "n_good", "pct_good",
    "n_acceptable", "pct_acceptable",
    "n_poor", "pct_poor",
    "mean_r2_L", "mean_r2_a", "mean_r2_b"
  ),
  value = c(
    nrow(df), n_total, n_no_checker, pct_no_checker,
    avg_n_patches,
    before["mean"], before["sd"], before["min"], before["max"], before["range"],
    after["mean"], after["sd"], after["min"], after["max"], after["range"],
    n_poor_after,
    get_q("good", "n"), get_q("good", "pct"),
    get_q("acceptable", "n"), get_q("acceptable", "pct"),
    get_q("poor", "n"), get_q("poor", "pct"),
    r2_L, r2_a, r2_b
  )
)

print(summary_df) # Print summary

write.csv(summary_df, "calibration_summary.csv", row.names = FALSE) # Write CSV
