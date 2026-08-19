# Colour Calibration Assessment

# Assesses calibration pipeline performance from correction_factors.csv

library(here)
library(dplyr)
library(ggplot2)
library(patchwork)

# Load data
df <- read.csv(here("validation", "colour_calibration", "calibration_performance", "correction_factors.csv"), stringsAsFactors = FALSE)

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

### FILTER OUT FAILED CALIBRATION IMAGES 
df_ok <- df[df$status == "calibrated", ]
cat(sprintf("\nRetained %d of %d images (calibration succeeded)\n", nrow(df_ok), nrow(df)))

# Create quality levels
quality_levels <- c("poor", "acceptable", "good") 

# Order quality as a factor, worst to best, for consistent plotting order
df_ok$quality <- factor(df_ok$quality, levels = quality_levels)

### PLOTS: dE spread before calibration, and after calibration by quality 

# shared minimal theme: no gridlines, external tick marks, no titles
no_grid_theme <- theme_classic(base_size = 12) +
  theme(
    legend.position = "none",
    panel.grid      = element_blank()
  )

# Panel A: dE_before across ALL calibrated images (no quality split) 
mean_before <- mean(df_ok$dE_before, na.rm = TRUE)
sd_before   <- sd(df_ok$dE_before,   na.rm = TRUE)
label_before <- sprintf("%.2f \u00B1 %.2f", mean_before, sd_before)
y_pos_before <- max(df_ok$dE_before, na.rm = TRUE) * 1.08

p_before <- ggplot(df_ok, aes(x = "All images", y = dE_before)) +
  geom_boxplot(width = 0.4, fill = "#C44E52", outlier.size = 0.8, outlier.alpha = 0.4) +
  annotate("text", x = 1, y = y_pos_before, label = label_before, size = 3.8) +
  labs(x = NULL, y = expression(Delta * "E")) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.12))) +
  no_grid_theme

# Panel B: dE_after split by quality (poor / acceptable / good) 
qual_n      <- table(df_ok$quality)
qual_labels <- setNames(paste0(names(qual_n), "\n(n=", as.integer(qual_n), ")"),
                        names(qual_n))

# per-quality mean +/- SD of dE_after, positioned just above each group's max
agg_mean <- tapply(df_ok$dE_after, df_ok$quality, mean, na.rm = TRUE)
agg_sd   <- tapply(df_ok$dE_after, df_ok$quality, sd,   na.rm = TRUE)
agg_max  <- tapply(df_ok$dE_after, df_ok$quality, max,  na.rm = TRUE)

stats_after <- data.frame(
  quality = factor(names(agg_mean), levels = quality_levels),
  label   = sprintf("%.2f \u00B1 %.2f", agg_mean, agg_sd),
  y_pos   = as.numeric(agg_max) * 1.08
)

p_after <- ggplot(df_ok, aes(x = quality, y = dE_after, fill = quality)) +
  geom_boxplot(width = 0.5, outlier.size = 0.8, outlier.alpha = 0.4) +
  geom_text(data = stats_after, aes(x = quality, y = y_pos, label = label),
            inherit.aes = FALSE, size = 3.8) +
  scale_fill_manual(values = c("poor" = "#C44E52", "acceptable" = "#DD8452", "good" = "#55A868")) +
  scale_x_discrete(labels = qual_labels[levels(df_ok$quality)]) +
  labs(x = NULL, y = expression(Delta * "E after calibration")) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.12))) +
  no_grid_theme

# Stitch together with patchwork, tagged A / B
combined_plot <- p_before + p_after +
  patchwork::plot_annotation(tag_levels = "A")

# Save alongside the other correction-factor outputs
ggsave(
  filename = here("validation", "colour_calibration", "calibration_performance", "dE_before_after_by_quality.png"),
  plot     = combined_plot,
  width    = 10, height = 5, dpi = 300
)

combined_plot

### END OF SCRIPT ###
