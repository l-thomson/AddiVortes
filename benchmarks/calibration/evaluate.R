# Nightly evaluation of the calibration output: rank ECDF difference
# plots with simultaneous confidence bands (Saeilynoja, Buerkner and
# Vehtari 2022) through the SBC package, and Geweke sample comparison
# plots. Reads the CSV files the full-size Rust tests write and puts the
# plots next to them.
#
# Usage: Rscript evaluate.R <calibration-dir>

library(SBC)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) >= 1) args[[1]] else "target/calibration"

ranks <- read.csv(file.path(dir, "sbc_ranks.csv"))
max_rank <- max(ranks$max_rank)
stats <- data.frame(
  sim_id = ave(seq_along(ranks$rank), ranks$quantity, FUN = seq_along),
  variable = ranks$quantity,
  rank = ranks$rank,
  max_rank = max_rank
)

p <- plot_ecdf_diff(stats)
ggsave(file.path(dir, "sbc_ecdf_diff.png"), p, width = 9, height = 6, dpi = 150)
p <- plot_rank_hist(stats)
ggsave(file.path(dir, "sbc_rank_hist.png"), p, width = 9, height = 6, dpi = 150)

geweke <- read.csv(file.path(dir, "geweke_samples.csv"))
p <- ggplot(geweke, aes(value, colour = simulator)) +
  stat_ecdf(pad = FALSE) +
  facet_wrap(~quantity, scales = "free_x") +
  labs(
    x = "value",
    y = "empirical CDF",
    title = "Geweke marginal-conditional against successive-conditional"
  )
ggsave(file.path(dir, "geweke_ecdf.png"), p, width = 9, height = 6, dpi = 150)

cat("wrote", file.path(dir, "sbc_ecdf_diff.png"), "\n")
cat("wrote", file.path(dir, "sbc_rank_hist.png"), "\n")
cat("wrote", file.path(dir, "geweke_ecdf.png"), "\n")
