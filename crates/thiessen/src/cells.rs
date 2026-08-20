//! Conjugate Gaussian cell means: sufficient statistics, the integrated
//! likelihood of a tessellation's cell structure, and the posterior draw of
//! the means.
//!
//! Observation i carries precision w_i = 1 / sigma_i^2 (the global 1 / sigma^2
//! for the Gaussian model), so the same code serves a per-observation
//! variance. With partial residuals r_i and prior mu_k ~ N(0, sigma_mu^2),
//! cell k with W_k = sum w_i and S_k = sum w_i r_i over its observations has
//!
//! ```text
//! ln p(r | T) = sum_k [ -ln(1 + W_k sigma_mu^2) / 2 + sigma_mu^2 S_k^2 / (2 (1 + W_k sigma_mu^2)) ]
//!               + terms that do not depend on T,
//! mu_k | r ~ N( sigma_mu^2 S_k / (1 + W_k sigma_mu^2), sigma_mu^2 / (1 + W_k sigma_mu^2) ).
//! ```
//!
//! Chipman, George and McCulloch (2010), s. 3.1, with the cell in place of
//! the leaf.

use crate::maths;
use crate::rng::{standard_normal, Rng};

/// Per-cell (n_k, W_k, S_k) accumulated in ascending observation order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CellStats {
    pub count: Vec<usize>,
    pub weight: Vec<f64>,
    pub sum: Vec<f64>,
}

impl CellStats {
    /// Statistics of `b` cells from the assignment, partial residuals and
    /// precisions.
    pub(crate) fn accumulate(
        cells: &[usize],
        residuals: &[f64],
        precision: &[f64],
        b: usize,
    ) -> Self {
        let mut count = vec![0_usize; b];
        let mut weight = vec![0.0; b];
        let mut sum = vec![0.0; b];
        for ((&cell, &r), &w) in cells.iter().zip(residuals).zip(precision) {
            count[cell] += 1;
            weight[cell] += w;
            sum[cell] += w * r;
        }
        Self { count, weight, sum }
    }

    /// Whether every cell holds at least one observation. Counted, not
    /// weighed: under prior-only sampling every precision is zero and the
    /// occupancy rule must not change.
    pub(crate) fn all_occupied(&self) -> bool {
        self.count.iter().all(|&c| c > 0)
    }

    /// The T-dependent part of the integrated log-likelihood.
    pub(crate) fn log_marginal(&self, sigma_mu_sq: f64) -> f64 {
        let mut total = 0.0;
        for (&w, &s) in self.weight.iter().zip(&self.sum) {
            let den = 1.0 + w * sigma_mu_sq;
            total += -0.5 * maths::ln(den) + sigma_mu_sq * s * s / (2.0 * den);
        }
        total
    }

    /// Posterior draw of every cell mean, ascending cell index, one
    /// standard normal per cell.
    pub(crate) fn draw_means(&self, sigma_mu_sq: f64, rng: &mut Rng) -> Vec<f64> {
        self.weight
            .iter()
            .zip(&self.sum)
            .map(|(&w, &s)| {
                let den = 1.0 + w * sigma_mu_sq;
                let mean = sigma_mu_sq * s / den;
                let var = sigma_mu_sq / den;
                mean + var.sqrt() * standard_normal(rng)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::chain_rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b}");
    }

    #[test]
    fn accumulate_and_occupancy() {
        let stats = CellStats::accumulate(&[0, 1, 1], &[1.0, 2.0, 3.0], &[2.0, 2.0, 2.0], 3);
        assert_eq!(stats.weight, vec![2.0, 4.0, 0.0]);
        assert_eq!(stats.sum, vec![2.0, 10.0, 0.0]);
        assert!(!stats.all_occupied());
        assert!(CellStats::accumulate(&[0, 1], &[1.0, 1.0], &[1.0, 1.0], 2).all_occupied());
    }

    #[test]
    fn log_marginal_matches_the_unweighted_form() {
        // With w_i = 1 / sigma^2: n_k = 4, S = 2, sigma^2 = 0.5, sigma_mu^2 = 0.25
        // gives 0.5 ln(sigma^2 / (n sigma_mu^2 + sigma^2)) + sigma_mu^2 S^2 /
        // (2 sigma^2 (n sigma_mu^2 + sigma^2)).
        let sigma_sq = 0.5;
        let stats = CellStats::accumulate(&[0; 4], &[0.5; 4], &[1.0 / sigma_sq; 4], 1);
        let expected = 0.5 * (sigma_sq / (4.0 * 0.25 + sigma_sq)).ln()
            + 0.25 * 4.0 / (2.0 * sigma_sq * (4.0 * 0.25 + sigma_sq));
        close(stats.log_marginal(0.25), expected, 1e-14);
    }

    /// ln N(r; 0, D^-1 + sigma_mu^2 Z Z^T) with D = diag(w) and Z the cell
    /// indicator matrix, by dense Cholesky. Shares no code with
    /// `log_marginal`.
    fn dense_log_density(r: &[f64], w: &[f64], cells: &[usize], sigma_mu_sq: f64) -> f64 {
        let n = r.len();
        let mut sigma = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut v = if cells[i] == cells[j] {
                    sigma_mu_sq
                } else {
                    0.0
                };
                if i == j {
                    v += 1.0 / w[i];
                }
                sigma[i * n + j] = v;
            }
        }
        let mut l = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = sigma[i * n + j];
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    l[i * n + i] = sum.sqrt();
                } else {
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
        }
        let mut u = r.to_vec();
        for i in 0..n {
            for k in 0..i {
                u[i] -= l[i * n + k] * u[k];
            }
            u[i] /= l[i * n + i];
        }
        let log_det: f64 = (0..n).map(|i| 2.0 * l[i * n + i].ln()).sum();
        let quad: f64 = u.iter().map(|v| v * v).sum();
        -0.5 * (n as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + quad)
    }

    #[test]
    fn log_marginal_difference_matches_a_dense_recomputation() {
        // log_marginal drops terms that do not depend on the assignment,
        // so differences between assignments equal the dense evaluation.
        use crate::rng::{chain_rng, standard_normal, uniform_index};
        let mut rng = chain_rng(17);
        for case in 0..60 {
            let n = 4 + uniform_index(9, &mut rng);
            let b = 1 + uniform_index(4, &mut rng);
            let sigma_mu_sq = 0.05 + 0.4 * (case as f64 / 60.0);
            let r: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
            let w: Vec<f64> = (0..n)
                .map(|_| (0.6 * standard_normal(&mut rng)).exp())
                .collect();
            let cells_a: Vec<usize> = (0..n).map(|_| uniform_index(b, &mut rng)).collect();
            let cells_b: Vec<usize> = (0..n).map(|_| uniform_index(b, &mut rng)).collect();
            let delta = CellStats::accumulate(&cells_a, &r, &w, b).log_marginal(sigma_mu_sq)
                - CellStats::accumulate(&cells_b, &r, &w, b).log_marginal(sigma_mu_sq);
            let dense = dense_log_density(&r, &w, &cells_a, sigma_mu_sq)
                - dense_log_density(&r, &w, &cells_b, sigma_mu_sq);
            close(delta, dense, 1e-9);
        }
    }

    #[test]
    fn posterior_mean_and_variance() {
        // n = 4, S = 2, sigma^2 = 0.5, sigma_mu^2 = 0.25: mean 1/3, var 1/12.
        let stats = CellStats::accumulate(&[0; 4], &[0.5; 4], &[2.0; 4], 1);
        let mut rng = chain_rng(1);
        let n = 40_000;
        let draws: Vec<f64> = (0..n)
            .map(|_| stats.draw_means(0.25, &mut rng)[0])
            .collect();
        let mean = draws.iter().sum::<f64>() / n as f64;
        let var = draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;
        close(mean, 1.0 / 3.0, 0.01);
        close(var, 1.0 / 12.0, 0.005);
    }
}
