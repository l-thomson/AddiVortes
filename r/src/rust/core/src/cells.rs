//! Conjugate cell families: the sufficient statistics of a tessellation's
//! cells, the integrated likelihood of its cell structure, and the
//! posterior draw of its cell values. An [`Ensemble`](crate::ensemble)
//! is generic over the family.
//!
//! # Gaussian cell means
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
//!
//! # Inverse-gamma cell variances
//!
//! Observation i carries a residual e_i and the product s_i of the other
//! variance tessellations at x_i, so e_i ~ N(0, v_k s_i) in cell k. With
//! prior v_k ~ Inv-Gamma(nu' / 2, nu' lambda' / 2), cell k with n_k
//! observations and E_k = sum e_i^2 / s_i has
//!
//! ```text
//! ln p(e | T) = sum_k [ lgamma((nu' + n_k) / 2) - lgamma(nu' / 2)
//!                      + (nu' / 2) ln(nu' lambda' / 2)
//!                      - ((nu' + n_k) / 2) ln((nu' lambda' + E_k) / 2) ]
//!               + terms that do not depend on T,
//! v_k | e ~ Inv-Gamma((nu' + n_k) / 2, (nu' lambda' + E_k) / 2).
//! ```
//!
//! The per-cell normaliser (nu' / 2) ln(nu' lambda' / 2) - lgamma(nu' / 2)
//! enters once per cell, so it appears in the acceptance ratio of every
//! move that changes the cell count. Pratola, Chipman, George and
//! McCulloch (2020), s. 3.2, with the cell in place of the leaf.

use crate::data::Data;
use crate::maths;
use crate::rng::{self, standard_normal, Rng};
use crate::tessellation::Tessellation;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::GaussianCells {}
    impl Sealed for super::InverseGammaCells {}
}

/// Per-cell sufficient statistics.
pub(crate) trait Stats {
    /// Whether every cell holds at least one observation. Counted, not
    /// weighed: under prior-only sampling every weight is zero and the
    /// occupancy rule must not change.
    fn all_occupied(&self) -> bool;
}

/// The per-observation inputs of one observation's contribution to its
/// tessellation's statistics, and the leave-one-out total of the ensemble.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Partial {
    /// The value entering the statistics.
    pub value: f64,
    /// The weight entering the statistics.
    pub weight: f64,
    /// The ensemble total with this tessellation's contribution removed.
    pub rest: f64,
}

/// A conjugate family of cell values with its combination rule across an
/// ensemble. Sealed: the two implementations are the crate's.
pub(crate) trait CellFamily: sealed::Sealed {
    type Stats: Stats;

    /// The observation's contribution to tessellation j's statistics, from
    /// the caller's input and weight, the ensemble total and the
    /// tessellation's own current value at the observation.
    fn partial(&self, input: f64, weight: f64, total: f64, own: f64) -> Partial;

    /// The ensemble total after tessellation j's value at the observation
    /// becomes `own`.
    fn total(&self, input: f64, partial: &Partial, own: f64) -> f64;

    /// Statistics of the tessellation's `b` cells from the assignment
    /// and the partials; `x` and the tessellation supply the offsets of
    /// the linear basis and are unused otherwise.
    fn accumulate(
        &self,
        x: &Data,
        tessellation: &Tessellation,
        cells: &[usize],
        partials: &[Partial],
        b: usize,
    ) -> Self::Stats;

    /// The T-dependent part of the integrated log-likelihood.
    fn log_marginal(
        &self,
        stats: &Self::Stats,
        #[cfg(test)] breakage: crate::broken::Breakage,
    ) -> f64;

    /// Posterior draw of every cell, ascending cell index: the values,
    /// and the row-major slopes of the linear basis (empty otherwise).
    fn draw(&self, stats: &Self::Stats, rng: &mut Rng) -> (Vec<f64>, Vec<f64>);

    /// The per-cell constant of the integrated likelihood, the term a
    /// fixture drops to mis-price the cell-count moves; zero for a family
    /// without one.
    #[cfg(test)]
    fn cell_normaliser(&self) -> f64;
}

/// Gaussian cell means under an additive ensemble; with `linear` the
/// cells carry (mu, beta) jointly, u = [1, x_A - c] the within-cell
/// design and N(0, sigma_mu^2 I) the prior on the whole coefficient
/// vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GaussianCells {
    pub sigma_mu_sq: f64,
    /// The linear cell basis ([`Basis::Linear`](crate::Basis)); reachable
    /// only through the `experimental` configuration surface.
    pub linear: bool,
}

/// Per-cell (n_k, W_k, S_k) accumulated in ascending observation order;
/// under the linear basis also the normal-equation blocks, per cell the
/// row-major q x q matrix A_k = sum w u u' and the q-vector
/// b_k = sum w r u with q = d + 1.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GaussianStats {
    pub count: Vec<usize>,
    pub weight: Vec<f64>,
    pub sum: Vec<f64>,
    q: usize,
    a: Vec<f64>,
    bvec: Vec<f64>,
}

impl Stats for GaussianStats {
    fn all_occupied(&self) -> bool {
        self.count.iter().all(|&c| c > 0)
    }
}

impl CellFamily for GaussianCells {
    type Stats = GaussianStats;

    fn partial(&self, input: f64, weight: f64, total: f64, own: f64) -> Partial {
        Partial {
            value: input - total + own,
            weight,
            rest: 0.0,
        }
    }

    fn total(&self, input: f64, partial: &Partial, own: f64) -> f64 {
        input - partial.value + own
    }

    fn accumulate(
        &self,
        x: &Data,
        tessellation: &Tessellation,
        cells: &[usize],
        partials: &[Partial],
        b: usize,
    ) -> GaussianStats {
        let mut count = vec![0_usize; b];
        let mut weight = vec![0.0; b];
        let mut sum = vec![0.0; b];
        let (q, mut a, mut bvec) = if self.linear {
            let q = tessellation.n_dims() + 1;
            (q, vec![0.0; b * q * q], vec![0.0; b * q])
        } else {
            (0, Vec::new(), Vec::new())
        };
        let mut u = vec![0.0; q];
        for (i, (&cell, p)) in cells.iter().zip(partials).enumerate() {
            count[cell] += 1;
            weight[cell] += p.weight;
            sum[cell] += p.weight * p.value;
            if self.linear {
                u[0] = 1.0;
                let row = x.row(i);
                let centre = tessellation.centre(cell);
                for (j, &dim) in tessellation.dims().iter().enumerate() {
                    u[j + 1] = row[dim] - centre[j];
                }
                for r in 0..q {
                    bvec[cell * q + r] += p.weight * p.value * u[r];
                    for c in 0..q {
                        a[cell * q * q + r * q + c] += p.weight * u[r] * u[c];
                    }
                }
            }
        }
        GaussianStats {
            count,
            weight,
            sum,
            q,
            a,
            bvec,
        }
    }

    fn log_marginal(
        &self,
        stats: &GaussianStats,
        #[cfg(test)] breakage: crate::broken::Breakage,
    ) -> f64 {
        if !self.linear {
            let mut total = 0.0;
            for (&w, &s) in stats.weight.iter().zip(&stats.sum) {
                let den = 1.0 + w * self.sigma_mu_sq;
                total += -0.5 * maths::ln(den) + self.sigma_mu_sq * s * s / (2.0 * den);
            }
            return total;
        }
        // Per cell -ln det(I + sigma_mu^2 A) / 2 + b' (Sigma0^-1 + A)^-1 b / 2,
        // the (d + 1)-dimensional form of the constant-basis expression.
        let q = stats.q;
        let mut total = 0.0;
        for cell in 0..stats.count.len() {
            let mut p_mat = stats.a[cell * q * q..(cell + 1) * q * q].to_vec();
            for r in 0..q {
                p_mat[r * q + r] += 1.0 / self.sigma_mu_sq;
            }
            let log_det_p = cholesky_in_place(&mut p_mat, q);
            let mut m = stats.bvec[cell * q..(cell + 1) * q].to_vec();
            cholesky_solve(&p_mat, q, &mut m);
            let fit: f64 = stats.bvec[cell * q..(cell + 1) * q]
                .iter()
                .zip(&m)
                .map(|(&b, &m)| b * m)
                .sum();
            let det_term = q as f64 * maths::ln(self.sigma_mu_sq) + log_det_p;
            #[cfg(test)]
            let det_term = if breakage == crate::broken::Breakage::DroppedTiltDeterminant {
                0.0
            } else {
                det_term
            };
            total += -0.5 * det_term + 0.5 * fit;
        }
        total
    }

    /// One standard normal per cell; under the linear basis q = d + 1
    /// standard normals per cell, coefficient order.
    fn draw(&self, stats: &GaussianStats, rng: &mut Rng) -> (Vec<f64>, Vec<f64>) {
        if !self.linear {
            let values = stats
                .weight
                .iter()
                .zip(&stats.sum)
                .map(|(&w, &s)| {
                    let den = 1.0 + w * self.sigma_mu_sq;
                    let mean = self.sigma_mu_sq * s / den;
                    let var = self.sigma_mu_sq / den;
                    mean + var.sqrt() * standard_normal(rng)
                })
                .collect();
            return (values, Vec::new());
        }
        let q = stats.q;
        let b = stats.count.len();
        let mut values = Vec::with_capacity(b);
        let mut slopes = Vec::with_capacity(b * (q - 1));
        for cell in 0..b {
            let mut p_mat = stats.a[cell * q * q..(cell + 1) * q * q].to_vec();
            for r in 0..q {
                p_mat[r * q + r] += 1.0 / self.sigma_mu_sq;
            }
            cholesky_in_place(&mut p_mat, q);
            let mut theta = stats.bvec[cell * q..(cell + 1) * q].to_vec();
            cholesky_solve(&p_mat, q, &mut theta);
            // theta = m + L^-T z: the upper-triangular solve of L' y = z.
            let mut z: Vec<f64> = (0..q).map(|_| standard_normal(rng)).collect();
            for r in (0..q).rev() {
                for c in r + 1..q {
                    z[r] -= p_mat[c * q + r] * z[c];
                }
                z[r] /= p_mat[r * q + r];
            }
            values.push(theta[0] + z[0]);
            for r in 1..q {
                slopes.push(theta[r] + z[r]);
            }
        }
        (values, slopes)
    }

    #[cfg(test)]
    fn cell_normaliser(&self) -> f64 {
        0.0
    }
}

/// The lower Cholesky factor of the row-major q x q symmetric positive
/// definite matrix, in place (the upper triangle is left stale); returns
/// ln det of the matrix.
fn cholesky_in_place(matrix: &mut [f64], q: usize) -> f64 {
    let mut log_det = 0.0;
    for j in 0..q {
        let mut d = matrix[j * q + j];
        for k in 0..j {
            d -= matrix[j * q + k] * matrix[j * q + k];
        }
        let root = d.sqrt();
        log_det += 2.0 * maths::ln(root);
        matrix[j * q + j] = root;
        for i in j + 1..q {
            let mut v = matrix[i * q + j];
            for k in 0..j {
                v -= matrix[i * q + k] * matrix[j * q + k];
            }
            matrix[i * q + j] = v / root;
        }
    }
    log_det
}

/// Solve (L L') x = rhs in place given the lower factor `l`.
fn cholesky_solve(l: &[f64], q: usize, rhs: &mut [f64]) {
    for r in 0..q {
        for c in 0..r {
            rhs[r] -= l[r * q + c] * rhs[c];
        }
        rhs[r] /= l[r * q + r];
    }
    for r in (0..q).rev() {
        for c in r + 1..q {
            rhs[r] -= l[c * q + r] * rhs[c];
        }
        rhs[r] /= l[r * q + r];
    }
}

/// Inverse-gamma cell variances under a multiplicative ensemble. Under
/// `prior_only` the likelihood is removed: every cell counts as empty in
/// the integrated likelihood and the draw, while occupancy still uses the
/// true counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct InverseGammaCells {
    pub nu: f64,
    pub lambda: f64,
    pub prior_only: bool,
}

/// Per-cell (n_k, E_k) accumulated in ascending observation order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InverseGammaStats {
    pub count: Vec<usize>,
    pub sum: Vec<f64>,
}

impl Stats for InverseGammaStats {
    fn all_occupied(&self) -> bool {
        self.count.iter().all(|&c| c > 0)
    }
}

impl InverseGammaCells {
    /// Shape and rate of cell k's posterior.
    fn posterior(&self, count: usize, sum: f64) -> (f64, f64) {
        if self.prior_only {
            (0.5 * self.nu, 0.5 * self.nu * self.lambda)
        } else {
            (
                0.5 * (self.nu + count as f64),
                0.5 * (self.nu * self.lambda + sum),
            )
        }
    }

    fn normaliser(&self) -> f64 {
        0.5 * self.nu * maths::ln(0.5 * self.nu * self.lambda) - maths::lgamma(0.5 * self.nu)
    }
}

impl CellFamily for InverseGammaCells {
    type Stats = InverseGammaStats;

    fn partial(&self, input: f64, _weight: f64, total: f64, own: f64) -> Partial {
        let rest = total / own;
        Partial {
            value: input,
            weight: 1.0 / rest,
            rest,
        }
    }

    fn total(&self, _input: f64, partial: &Partial, own: f64) -> f64 {
        partial.rest * own
    }

    fn accumulate(
        &self,
        _x: &Data,
        _tessellation: &Tessellation,
        cells: &[usize],
        partials: &[Partial],
        b: usize,
    ) -> InverseGammaStats {
        let mut count = vec![0_usize; b];
        let mut sum = vec![0.0; b];
        for (&cell, p) in cells.iter().zip(partials) {
            count[cell] += 1;
            sum[cell] += p.weight * p.value * p.value;
        }
        InverseGammaStats { count, sum }
    }

    fn log_marginal(
        &self,
        stats: &InverseGammaStats,
        #[cfg(test)] _breakage: crate::broken::Breakage,
    ) -> f64 {
        let normaliser = self.normaliser();
        let mut total = 0.0;
        for (&n, &e) in stats.count.iter().zip(&stats.sum) {
            let (shape, rate) = self.posterior(n, e);
            total += maths::lgamma(shape) - shape * maths::ln(rate) + normaliser;
        }
        total
    }

    /// One gamma draw per cell.
    fn draw(&self, stats: &InverseGammaStats, rng: &mut Rng) -> (Vec<f64>, Vec<f64>) {
        let values = stats
            .count
            .iter()
            .zip(&stats.sum)
            .map(|(&n, &e)| {
                let (shape, rate) = self.posterior(n, e);
                1.0 / rng::gamma(shape, 1.0 / rate, rng)
            })
            .collect();
        (values, Vec::new())
    }

    #[cfg(test)]
    fn cell_normaliser(&self) -> f64 {
        self.normaliser()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{chain_rng, uniform_index};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b}");
    }

    /// A design and tessellation for the constant-basis calls, which
    /// ignore both.
    fn any_ctx(n: usize) -> (Data, Tessellation) {
        let rows: Vec<[f64; 1]> = (0..n.max(2)).map(|i| [i as f64]).collect();
        let t = Tessellation {
            centres: vec![0.0],
            dims: vec![0],
            mus: vec![0.0],
            betas: Vec::new(),
        };
        (Data::from_rows(&rows).unwrap(), t)
    }

    fn gaussian_partials(residuals: &[f64], precision: &[f64]) -> Vec<Partial> {
        residuals
            .iter()
            .zip(precision)
            .map(|(&r, &w)| Partial {
                value: r,
                weight: w,
                rest: 0.0,
            })
            .collect()
    }

    #[test]
    fn gaussian_accumulate_and_occupancy() {
        let ctx = any_ctx(8);
        let family = GaussianCells {
            sigma_mu_sq: 0.25,
            linear: false,
        };
        let stats = family.accumulate(
            &ctx.0,
            &ctx.1,
            &[0, 1, 1],
            &gaussian_partials(&[1.0, 2.0, 3.0], &[2.0, 2.0, 2.0]),
            3,
        );
        assert_eq!(stats.weight, vec![2.0, 4.0, 0.0]);
        assert_eq!(stats.sum, vec![2.0, 10.0, 0.0]);
        assert!(!stats.all_occupied());
        assert!(family
            .accumulate(
                &ctx.0,
                &ctx.1,
                &[0, 1],
                &gaussian_partials(&[1.0, 1.0], &[1.0, 1.0]),
                2
            )
            .all_occupied());
    }

    #[test]
    fn gaussian_partial_and_total_use_the_backfitting_order() {
        // r = y - F + mu and F' = y - r + mu', in that order of operations.
        let family = GaussianCells {
            sigma_mu_sq: 0.25,
            linear: false,
        };
        let p = family.partial(0.7, 3.0, 0.2, 0.05);
        assert_eq!(p.value, 0.7 - 0.2 + 0.05);
        assert_eq!(p.weight, 3.0);
        assert_eq!(family.total(0.7, &p, 0.1), 0.7 - p.value + 0.1);
    }

    #[test]
    fn gaussian_log_marginal_matches_the_unweighted_form() {
        let ctx = any_ctx(8);
        // With w_i = 1 / sigma^2: n_k = 4, S = 2, sigma^2 = 0.5, sigma_mu^2 = 0.25
        // gives 0.5 ln(sigma^2 / (n sigma_mu^2 + sigma^2)) + sigma_mu^2 S^2 /
        // (2 sigma^2 (n sigma_mu^2 + sigma^2)).
        let sigma_sq = 0.5;
        let family = GaussianCells {
            sigma_mu_sq: 0.25,
            linear: false,
        };
        let stats = family.accumulate(
            &ctx.0,
            &ctx.1,
            &[0; 4],
            &gaussian_partials(&[0.5; 4], &[1.0 / sigma_sq; 4]),
            1,
        );
        let expected = 0.5 * (sigma_sq / (4.0 * 0.25 + sigma_sq)).ln()
            + 0.25 * 4.0 / (2.0 * sigma_sq * (4.0 * 0.25 + sigma_sq));
        close(
            family.log_marginal(&stats, crate::broken::Breakage::None),
            expected,
            1e-14,
        );
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
    fn gaussian_log_marginal_difference_matches_a_dense_recomputation() {
        let ctx = any_ctx(8);
        // log_marginal drops terms that do not depend on the assignment,
        // so differences between assignments equal the dense evaluation.
        let mut rng = chain_rng(17);
        for case in 0..60 {
            let n = 4 + uniform_index(9, &mut rng);
            let b = 1 + uniform_index(4, &mut rng);
            let family = GaussianCells {
                sigma_mu_sq: 0.05 + 0.4 * (case as f64 / 60.0),
                linear: false,
            };
            let r: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
            let w: Vec<f64> = (0..n)
                .map(|_| (0.6 * standard_normal(&mut rng)).exp())
                .collect();
            let cells_a: Vec<usize> = (0..n).map(|_| uniform_index(b, &mut rng)).collect();
            let cells_b: Vec<usize> = (0..n).map(|_| uniform_index(b, &mut rng)).collect();
            let partials = gaussian_partials(&r, &w);
            let delta = family.log_marginal(
                &family.accumulate(&ctx.0, &ctx.1, &cells_a, &partials, b),
                crate::broken::Breakage::None,
            ) - family.log_marginal(
                &family.accumulate(&ctx.0, &ctx.1, &cells_b, &partials, b),
                crate::broken::Breakage::None,
            );
            let dense = dense_log_density(&r, &w, &cells_a, family.sigma_mu_sq)
                - dense_log_density(&r, &w, &cells_b, family.sigma_mu_sq);
            close(delta, dense, 1e-9);
        }
    }

    #[test]
    fn gaussian_posterior_mean_and_variance() {
        let ctx = any_ctx(8);
        // n = 4, S = 2, sigma^2 = 0.5, sigma_mu^2 = 0.25: mean 1/3, var 1/12.
        let family = GaussianCells {
            sigma_mu_sq: 0.25,
            linear: false,
        };
        let stats = family.accumulate(
            &ctx.0,
            &ctx.1,
            &[0; 4],
            &gaussian_partials(&[0.5; 4], &[2.0; 4]),
            1,
        );
        let mut rng = chain_rng(1);
        let n = 40_000;
        let draws: Vec<f64> = (0..n).map(|_| family.draw(&stats, &mut rng).0[0]).collect();
        let mean = draws.iter().sum::<f64>() / n as f64;
        let var = draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;
        close(mean, 1.0 / 3.0, 0.01);
        close(var, 1.0 / 12.0, 0.005);
    }

    fn ig_family() -> InverseGammaCells {
        InverseGammaCells {
            nu: 3.0,
            lambda: 0.4,
            prior_only: false,
        }
    }

    #[test]
    fn inverse_gamma_partial_and_total_divide_and_multiply() {
        let family = ig_family();
        let p = family.partial(0.3, 0.0, 6.0, 2.0);
        close(p.rest, 3.0, 1e-15);
        close(p.weight, 1.0 / 3.0, 1e-15);
        assert_eq!(p.value, 0.3);
        close(family.total(0.3, &p, 0.5), 1.5, 1e-15);
    }

    #[test]
    fn inverse_gamma_accumulate_and_occupancy() {
        let ctx = any_ctx(8);
        let family = ig_family();
        let partials = [
            Partial {
                value: 1.0,
                weight: 2.0,
                rest: 0.5,
            },
            Partial {
                value: 2.0,
                weight: 0.5,
                rest: 2.0,
            },
            Partial {
                value: 3.0,
                weight: 1.0,
                rest: 1.0,
            },
        ];
        let stats = family.accumulate(&ctx.0, &ctx.1, &[0, 1, 1], &partials, 3);
        assert_eq!(stats.count, vec![1, 2, 0]);
        assert_eq!(stats.sum, vec![2.0, 2.0 + 9.0, 0.0]);
        assert!(!stats.all_occupied());
    }

    /// ln p(e | T) = sum_k ln integral prod_{i in k} N(e_i; 0, v s_i)
    /// Inv-Gamma(v; nu / 2, nu lambda / 2) dv, each cell integral by the
    /// trapezium rule over t = ln v. Shares no code with `log_marginal`.
    fn quadrature_log_density(
        e: &[f64],
        s: &[f64],
        cells: &[usize],
        b: usize,
        nu: f64,
        lambda: f64,
    ) -> f64 {
        let (a, rate) = (0.5 * nu, 0.5 * nu * lambda);
        let log_prior_const = a * rate.ln() - libm::lgamma(a);
        let steps = 20_000;
        let (lo, hi) = (lambda.ln() - 14.0, lambda.ln() + 14.0);
        let h = (hi - lo) / steps as f64;
        let mut total = 0.0;
        for k in 0..b {
            let members: Vec<usize> = (0..e.len()).filter(|&i| cells[i] == k).collect();
            let mut log_terms = Vec::with_capacity(steps + 1);
            for step in 0..=steps {
                let t = lo + h * step as f64;
                let v = t.exp();
                // ln prior density in v times the Jacobian v, plus the likelihood.
                let mut lp = log_prior_const - (a + 1.0) * t - rate / v + t;
                for &i in &members {
                    let var = v * s[i];
                    lp += -0.5 * ((2.0 * std::f64::consts::PI * var).ln() + e[i] * e[i] / var);
                }
                log_terms.push(lp);
            }
            let max = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = log_terms.iter().map(|lp| (lp - max).exp()).sum();
            total += max + (sum * h).ln();
        }
        total
    }

    #[test]
    fn inverse_gamma_log_marginal_difference_matches_quadrature() {
        let ctx = any_ctx(8);
        let mut rng = chain_rng(23);
        for case in 0..40 {
            let n = 3 + uniform_index(8, &mut rng);
            let b = 1 + uniform_index(3, &mut rng);
            let family = InverseGammaCells {
                nu: 2.5 + 0.1 * case as f64,
                lambda: 0.2 + 0.02 * case as f64,
                prior_only: false,
            };
            let e: Vec<f64> = (0..n).map(|_| 0.5 * standard_normal(&mut rng)).collect();
            let s: Vec<f64> = (0..n)
                .map(|_| (0.5 * standard_normal(&mut rng)).exp())
                .collect();
            let partials: Vec<Partial> = e
                .iter()
                .zip(&s)
                .map(|(&e, &s)| Partial {
                    value: e,
                    weight: 1.0 / s,
                    rest: s,
                })
                .collect();
            let cells_a: Vec<usize> = (0..n).map(|_| uniform_index(b, &mut rng)).collect();
            let cells_b: Vec<usize> = (0..n).map(|_| uniform_index(b, &mut rng)).collect();
            let delta = family.log_marginal(
                &family.accumulate(&ctx.0, &ctx.1, &cells_a, &partials, b),
                crate::broken::Breakage::None,
            ) - family.log_marginal(
                &family.accumulate(&ctx.0, &ctx.1, &cells_b, &partials, b),
                crate::broken::Breakage::None,
            );
            let reference = quadrature_log_density(&e, &s, &cells_a, b, family.nu, family.lambda)
                - quadrature_log_density(&e, &s, &cells_b, b, family.nu, family.lambda);
            close(delta, reference, 1e-6);
        }
    }

    #[test]
    fn inverse_gamma_log_marginal_closed_form() {
        let ctx = any_ctx(8);
        // nu = 3, lambda = 0.4; cell 0 holds one observation with e^2 / s = 2,
        // cell 1 holds two with e^2 / s = 0.5 each.
        let family = ig_family();
        let partial = |value: f64| Partial {
            value,
            weight: 0.5,
            rest: 2.0,
        };
        let partials = [partial(2.0), partial(1.0), partial(1.0)];
        let stats = family.accumulate(&ctx.0, &ctx.1, &[0, 1, 1], &partials, 2);
        assert_eq!(stats.sum, vec![2.0, 1.0]);
        let normaliser = 1.5 * 0.6_f64.ln() - libm::lgamma(1.5);
        let cell = |n: f64, e: f64| {
            libm::lgamma(0.5 * (3.0 + n)) - 0.5 * (3.0 + n) * (0.5 * (1.2 + e)).ln() + normaliser
        };
        close(family.cell_normaliser(), normaliser, 1e-14);
        close(
            family.log_marginal(&stats, crate::broken::Breakage::None),
            cell(1.0, 2.0) + cell(2.0, 1.0),
            1e-12,
        );
        // Under prior_only every cell contributes zero.
        let prior = InverseGammaCells {
            prior_only: true,
            ..family
        };
        close(
            prior.log_marginal(&stats, crate::broken::Breakage::None),
            0.0,
            1e-12,
        );
    }

    #[test]
    fn inverse_gamma_posterior_mean() {
        let ctx = any_ctx(8);
        // nu = 3, lambda = 0.4, n = 4, E = 2: Inv-Gamma(3.5, 1.6), mean 1.6 / 2.5.
        let family = ig_family();
        let partials = [Partial {
            value: 1.0,
            weight: 0.5,
            rest: 2.0,
        }; 4];
        let stats = family.accumulate(&ctx.0, &ctx.1, &[0; 4], &partials, 1);
        assert_eq!(stats.sum, vec![2.0]);
        let mut rng = chain_rng(2);
        let n = 100_000;
        let mean = (0..n)
            .map(|_| family.draw(&stats, &mut rng).0[0])
            .sum::<f64>()
            / n as f64;
        close(mean, 1.6 / 2.5, 0.01);
        let prior = InverseGammaCells {
            prior_only: true,
            ..family
        };
        // Inv-Gamma(1.5, 0.6) has no finite variance; its reciprocal is
        // Gamma(1.5, rate 0.6) with mean 2.5.
        let mean = (0..n)
            .map(|_| 1.0 / prior.draw(&stats, &mut rng).0[0])
            .sum::<f64>()
            / n as f64;
        close(mean, 2.5, 0.05);
    }

    mod linear {
        use super::*;

        fn linear_family() -> GaussianCells {
            GaussianCells {
                sigma_mu_sq: 0.25,
                linear: true,
            }
        }

        /// One cell on one dimension centred at 0.4.
        fn one_cell() -> Tessellation {
            Tessellation {
                centres: vec![0.4],
                dims: vec![0],
                mus: vec![0.0],
                betas: vec![0.0],
            }
        }

        #[test]
        fn rows_at_the_centre_match_the_constant_marginal() {
            // Every offset zero: the slope block is prior-only and the
            // integrated likelihood reduces to the constant expression.
            let x = Data::from_rows(&[[0.4], [0.4], [0.4]]).unwrap();
            let t = one_cell();
            let partials = gaussian_partials(&[0.6, -0.2, 0.1], &[2.0, 1.0, 3.0]);
            let cells = [0, 0, 0];
            let linear = linear_family();
            let constant = GaussianCells {
                sigma_mu_sq: 0.25,
                linear: false,
            };
            let a = linear.log_marginal(
                &linear.accumulate(&x, &t, &cells, &partials, 1),
                crate::broken::Breakage::None,
            );
            let b = constant.log_marginal(
                &constant.accumulate(&x, &t, &cells, &partials, 1),
                crate::broken::Breakage::None,
            );
            close(a, b, 1e-12);
        }

        #[test]
        fn the_marginal_matches_the_closed_form() {
            // Two observations, u = (1, x - 0.4), unit weights.
            let x = Data::from_rows(&[[0.9], [-0.1]]).unwrap();
            let t = one_cell();
            let partials = gaussian_partials(&[1.0, 0.2], &[1.0, 1.0]);
            let cells = [0, 0];
            let family = linear_family();
            let stats = family.accumulate(&x, &t, &cells, &partials, 1);
            // A = [[2, 0], [0, 0.5]], b = (1.2, 0.4).
            let sigma = 0.25_f64;
            let p11 = 2.0 + 1.0 / sigma;
            let p22 = 0.5 + 1.0 / sigma;
            let expected = -0.5 * ((sigma * p11).ln() + (sigma * p22).ln())
                + 0.5 * (1.2 * 1.2 / p11 + 0.4 * 0.4 / p22);
            close(
                family.log_marginal(&stats, crate::broken::Breakage::None),
                expected,
                1e-12,
            );
        }

        #[test]
        fn the_joint_draw_matches_the_conjugate_posterior() {
            let x = Data::from_rows(&[[0.9], [-0.1], [0.15], [0.65]]).unwrap();
            let t = one_cell();
            let partials = gaussian_partials(&[1.0, 0.2, 0.4, 0.9], &[1.0, 2.0, 1.0, 1.0]);
            let cells = [0, 0, 0, 0];
            let family = linear_family();
            let stats = family.accumulate(&x, &t, &cells, &partials, 1);
            // The closed-form posterior mean, from the normal equations.
            let offsets = [0.5, -0.5, -0.25, 0.25];
            let values = [1.0, 0.2, 0.4, 0.9];
            let weights = [1.0, 2.0, 1.0, 1.0];
            let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for i in 0..4 {
                a11 += weights[i];
                a12 += weights[i] * offsets[i];
                a22 += weights[i] * offsets[i] * offsets[i];
                b1 += weights[i] * values[i];
                b2 += weights[i] * values[i] * offsets[i];
            }
            let (p11, p12, p22) = (a11 + 4.0, a12, a22 + 4.0);
            let det = p11 * p22 - p12 * p12;
            let mean = [(p22 * b1 - p12 * b2) / det, (p11 * b2 - p12 * b1) / det];
            let cov = [p22 / det, -p12 / det, p11 / det];

            let mut rng = rng::chain_rng(11);
            let n = 200_000;
            let (mut m0, mut m1) = (0.0, 0.0);
            let (mut v0, mut v1, mut c01) = (0.0, 0.0, 0.0);
            for _ in 0..n {
                let (values, slopes) = family.draw(&stats, &mut rng);
                m0 += values[0];
                m1 += slopes[0];
                v0 += values[0] * values[0];
                v1 += slopes[0] * slopes[0];
                c01 += values[0] * slopes[0];
            }
            let n = n as f64;
            let (m0, m1) = (m0 / n, m1 / n);
            close(m0, mean[0], 5e-3);
            close(m1, mean[1], 5e-3);
            close(v0 / n - m0 * m0, cov[0], 5e-3);
            close(c01 / n - m0 * m1, cov[1], 5e-3);
            close(v1 / n - m1 * m1, cov[2], 5e-3);
        }
    }
}
