# abalone_morphometrics — model performance assessment
# Input:  abalone_measurements.csv

library(readxl)
library(dplyr)
library(ggplot2)
library(patchwork)   
library(here)

# Load 
dat <- read_excel("abalone_measurements.csv") |>
  rename(manual   = start_length_mm,
         detected = detected_length,
         flag     = `check?`) |>
  mutate(
    diff     = detected - manual,
    abs_err  = abs(diff),
    mean_val = (manual + detected) / 2,
    flagged  = flag != "0"
  )

# Clean subset: flagged rows removed
dat_clean <- filter(dat, !flagged)

# Metrics function
metrics <- function(d, label) {
  cat("\n──", label, "── n =", nrow(d), "\n")
  cat("  MAE  :", round(mean(d$abs_err), 3), "mm\n")
  cat("  RMSE :", round(sqrt(mean(d$diff^2)), 3), "mm\n")
  cat("  Bias :", round(mean(d$diff), 3), "mm  (Python - manual)\n")
  cat("  SD   :", round(sd(d$diff), 3), "mm\n")
  cat("  LoA  :", round(mean(d$diff) - 1.96 * sd(d$diff), 2),
      "to", round(mean(d$diff) + 1.96 * sd(d$diff), 2), "mm\n")
}

metrics(dat,       "All data")
metrics(dat_clean, "Flagged rows excluded")

# Bland-Altman plot function 
ba_plot <- function(d, title) {
  bias   <- mean(d$diff)
  sd_d   <- sd(d$diff)
  loa_lo <- bias - 1.96 * sd_d
  loa_hi <- bias + 1.96 * sd_d

  ggplot(d, aes(x = mean_val, y = diff)) +
    geom_point(alpha = 0.35, size = 1.2, colour = "#2C6E8E") +
    geom_hline(yintercept = bias,   colour = "#B23A48", linewidth = 0.9) +
    geom_hline(yintercept = loa_hi, colour = "#B23A48", linewidth = 0.7, linetype = "dashed") +
    geom_hline(yintercept = loa_lo, colour = "#B23A48", linewidth = 0.7, linetype = "dashed") +
    geom_hline(yintercept = 0,      colour = "grey50",  linewidth = 0.5, linetype = "dotted") +
    annotate("text", x = Inf, y = bias,   hjust = 1.1, vjust = -0.4,
             label = paste0("Bias = ", round(bias, 2), " mm"),   size = 3, colour = "#B23A48") +
    annotate("text", x = Inf, y = loa_hi, hjust = 1.1, vjust = -0.4,
             label = paste0("+1.96 SD = ", round(loa_hi, 2), " mm"), size = 3, colour = "#B23A48") +
    annotate("text", x = Inf, y = loa_lo, hjust = 1.1, vjust =  1.3,
             label = paste0("-1.96 SD = ", round(loa_lo, 2), " mm"), size = 3, colour = "#B23A48") +
    labs(title = title,
         x = "Mean of manual and detected length (mm)",
         y = "Difference: detected - manual (mm)") +
    coord_cartesian(ylim = c(-30, 30)) +
    theme_bw(base_size = 11) +
    theme(plot.title = element_text(size = 11, face = "bold"))
}

# Build and save plots 
p1 <- ba_plot(dat,       paste0("All data  (n = ", nrow(dat), ")"))
p2 <- ba_plot(dat_clean, paste0("Flagged rows excluded  (n = ", nrow(dat_clean), ")"))

combined <- p1 + p2 + plot_layout(ncol = 2)

ggsave(here("validation", "morphometrics","bland_altman_comparison.png"), combined, width = 12, height = 5, dpi = 300)

# Absolute error summary
print(quantile(dat$abs_err, probs = c(0, 0.25, 0.5, 0.75, 0.90, 0.95, 1)))

thresholds <- c(1, 2, 3, 5, 10, 15)
pct <- sapply(thresholds, function(t) mean(dat$abs_err <= t) * 100)
print(data.frame(threshold_mm = thresholds, within_pct = round(pct, 1)))
