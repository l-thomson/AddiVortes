//! The affine maps between the caller's scale and the sampler's scaled space,
//! and the prior calibration computed once at fit.
//!
//! Scaled space: y on [-0.5, 0.5] over its training range; every Euclidean
//! column of X min-max scaled to [-0.5, 0.5] over its training range;
//! columns of the other metrics left on the caller's scale.

use crate::data::Data;
use crate::error::{Error, Result};
use crate::geometry::Geometry;
use crate::maths;

/// The fitted scaling, stored on the fitted model and applied to every
/// prediction input. Deserialisation validates lengths and ranges.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "ScalerParts")]
pub struct Scaler {
    y_min: f64,
    y_max: f64,
    x_min: Vec<f64>,
    x_max: Vec<f64>,
}

impl Scaler {
    /// The identity maps: every training range is [-0.5, 0.5], so scaled
    /// and caller space coincide. Used by `Sampler::pinned_prior`.
    pub(crate) fn identity(p: usize) -> Self {
        Self {
            y_min: -0.5,
            y_max: 0.5,
            x_min: vec![-0.5; p],
            x_max: vec![0.5; p],
        }
    }

    /// Fit on validated data; returns the scaler, scaled X and scaled y.
    pub(crate) fn fit(x: &Data, y: &[f64], geometry: &Geometry) -> (Self, Data, Vec<f64>) {
        let (y_min, y_max) = min_max(y.iter().copied());
        let scaler = Self::with_y_range(x, y_min, y_max, geometry);
        let x_scaled = scaler.scale_x(x);
        let y_scaled = y.iter().map(|&v| scaler.scale_y(v)).collect();
        (scaler, x_scaled, y_scaled)
    }

    /// Fit the covariate maps on validated data with the identity map on
    /// the response (training range [-0.5, 0.5]); returns the scaler and
    /// scaled X. Used by the probit model, whose latent response is on a
    /// fixed scale.
    pub(crate) fn fit_x(x: &Data, geometry: &Geometry) -> (Self, Data) {
        let scaler = Self::with_y_range(x, -0.5, 0.5, geometry);
        let x_scaled = scaler.scale_x(x);
        (scaler, x_scaled)
    }

    /// Columns the geometry leaves unscaled take the identity range
    /// [-0.5, 0.5].
    fn with_y_range(x: &Data, y_min: f64, y_max: f64, geometry: &Geometry) -> Self {
        let n = x.n_rows();
        let p = x.n_cols();
        let mut x_min = vec![-0.5; p];
        let mut x_max = vec![0.5; p];
        for col in (0..p).filter(|&col| geometry.scaled(col)) {
            let (lo, hi) = min_max((0..n).map(|r| x.values()[r * p + col]));
            x_min[col] = lo;
            x_max[col] = hi;
        }
        Self {
            y_min,
            y_max,
            x_min,
            x_max,
        }
    }

    /// Scale X columnwise; values outside the training range map outside
    /// [-0.5, 0.5] and are not clamped.
    pub(crate) fn scale_x(&self, x: &Data) -> Data {
        let p = self.x_min.len();
        let values = x
            .values()
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let col = i % p;
                (v - self.x_min[col]) / (self.x_max[col] - self.x_min[col]) - 0.5
            })
            .collect();
        Data::new(values, x.n_rows(), p).expect("shape preserved")
    }

    /// Caller scale to scaled space.
    pub(crate) fn scale_y(&self, v: f64) -> f64 {
        (v - self.y_min) / (self.y_max - self.y_min) - 0.5
    }

    /// Scaled space to caller scale.
    pub(crate) fn unscale_y(&self, v: f64) -> f64 {
        (v + 0.5) * (self.y_max - self.y_min) + self.y_min
    }

    /// Training range of the response, y_max - y_min.
    pub fn y_range(&self) -> f64 {
        self.y_max - self.y_min
    }

    /// Minimum of the training response.
    pub fn y_min(&self) -> f64 {
        self.y_min
    }

    /// Maximum of the training response.
    pub fn y_max(&self) -> f64 {
        self.y_max
    }

    /// Per-column lower end of the range mapped to -0.5: the training
    /// minimum of a Euclidean column, -0.5 for a column the metric leaves
    /// unscaled.
    pub fn x_min(&self) -> &[f64] {
        &self.x_min
    }

    /// Per-column upper end of the range mapped to 0.5: the training
    /// maximum of a Euclidean column, 0.5 for a column the metric leaves
    /// unscaled.
    pub fn x_max(&self) -> &[f64] {
        &self.x_max
    }

    /// Number of columns p.
    pub fn n_cols(&self) -> usize {
        self.x_min.len()
    }
}

#[derive(serde::Deserialize)]
struct ScalerParts {
    y_min: f64,
    y_max: f64,
    x_min: Vec<f64>,
    x_max: Vec<f64>,
}

impl TryFrom<ScalerParts> for Scaler {
    type Error = Error;

    fn try_from(parts: ScalerParts) -> Result<Self> {
        let bad = |reason: &str| Error::InvalidSavedModel {
            reason: reason.into(),
        };
        if parts.x_min.len() != parts.x_max.len() {
            return Err(bad("scaler x_min and x_max differ in length"));
        }
        if !(parts.y_min.is_finite() && parts.y_max.is_finite() && parts.y_min < parts.y_max) {
            return Err(bad("scaler y range must be finite with y_min < y_max"));
        }
        for (lo, hi) in parts.x_min.iter().zip(&parts.x_max) {
            if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                return Err(bad("scaler x ranges must be finite with x_min < x_max"));
            }
        }
        Ok(Self {
            y_min: parts.y_min,
            y_max: parts.y_max,
            x_min: parts.x_min,
            x_max: parts.x_max,
        })
    }
}

fn min_max(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in values {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

/// sigma_mu^2 = (0.5 / (k sqrt m))^2, scaled space (Stone and Gosling 2025,
/// s. 2.3.2; Chipman, George and McCulloch 2010, s. 2.2.3 with the
/// response on [-0.5, 0.5]).
pub(crate) fn sigma_mu_sq(k: f64, m: usize) -> f64 {
    let s = 0.5 / (k * (m as f64).sqrt());
    s * s
}

/// sigma_mu^2 = (3 / (k sqrt m))^2 on the latent scale of the probit model
/// (Chipman, George and McCulloch 2010, s. 4: the latent mean is confined
/// to [-3, 3] with prior probability set by k; BART `pbart`).
pub(crate) fn probit_sigma_mu_sq(k: f64, m: usize) -> f64 {
    let s = 3.0 / (k * (m as f64).sqrt());
    s * s
}

/// The (nu', lambda') of one of m' variance tessellations such that the
/// product of m' independent Inv-Gamma(nu' / 2, nu' lambda' / 2) cell
/// values has the mean nu lambda / (nu - 2) of the Gaussian model's
/// sigma^2 ~ nu lambda / chi^2_nu: lambda' = lambda^(1 / m') and
/// nu' = 2 / (1 - (1 - 2 / nu)^(1 / m')). Requires nu > 2.
pub(crate) fn variance_cell_prior(nu: f64, lambda: f64, m_var: usize) -> (f64, f64) {
    let root = 1.0 / m_var as f64;
    let nu_prime = 2.0 / (1.0 - libm::pow(1.0 - 2.0 / nu, root));
    (nu_prime, libm::pow(lambda, root))
}

/// sigma_hat for the sigma^2 prior: the residual standard deviation of the
/// least-squares fit of scaled y on the scaled design with intercept when
/// n > p + 1, the normal equations are well conditioned and the residual
/// standard deviation is positive; otherwise the sample standard deviation
/// of scaled y (upstream `InitialSigma = "Linear"`). A zero sigma_hat
/// would degenerate the prior at sigma^2 = 0.
pub(crate) fn sigma_hat(x_scaled: &Data, y_scaled: &[f64]) -> f64 {
    ols_residual_sd(x_scaled, y_scaled).unwrap_or_else(|| sample_sd(y_scaled))
}

/// lambda of the prior sigma^2 ~ nu lambda / chi^2_nu such that
/// Pr(sigma < sigma_hat) = q: lambda = sigma_hat^2 chi^2_nu(1 - q) / nu.
pub(crate) fn calibrate_lambda(nu: f64, q: f64, sigma_hat: f64) -> f64 {
    sigma_hat * sigma_hat * maths::chi2_quantile(1.0 - q, nu) / nu
}

fn ols_residual_sd(x: &Data, y: &[f64]) -> Option<f64> {
    let n = x.n_rows();
    let p1 = x.n_cols() + 1;
    if n <= p1 {
        return None;
    }
    let design = |r: usize, j: usize| -> f64 {
        if j == 0 {
            1.0
        } else {
            x.row(r)[j - 1]
        }
    };
    let mut a = vec![0.0_f64; p1 * p1];
    let mut b = vec![0.0_f64; p1];
    for (r, &y_r) in y.iter().enumerate() {
        for i in 0..p1 {
            let di = design(r, i);
            b[i] += di * y_r;
            for j in i..p1 {
                a[i * p1 + j] += di * design(r, j);
            }
        }
    }
    for i in 0..p1 {
        for j in 0..i {
            a[i * p1 + j] = a[j * p1 + i];
        }
    }
    let max_diag = (0..p1).fold(0.0_f64, |acc, i| acc.max(a[i * p1 + i].abs()));
    let mut l = vec![0.0_f64; p1 * p1];
    for i in 0..p1 {
        for j in 0..=i {
            let mut sum = a[i * p1 + j];
            for k in 0..j {
                sum -= l[i * p1 + k] * l[j * p1 + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 1e-10 * max_diag {
                    return None;
                }
                l[i * p1 + i] = sum.sqrt();
            } else {
                l[i * p1 + j] = sum / l[j * p1 + j];
            }
        }
    }
    let mut beta = b;
    for i in 0..p1 {
        for k in 0..i {
            beta[i] -= l[i * p1 + k] * beta[k];
        }
        beta[i] /= l[i * p1 + i];
    }
    for i in (0..p1).rev() {
        for k in (i + 1)..p1 {
            beta[i] -= l[k * p1 + i] * beta[k];
        }
        beta[i] /= l[i * p1 + i];
    }
    let mut rss = 0.0_f64;
    for (r, &y_r) in y.iter().enumerate() {
        let fitted: f64 = beta.iter().enumerate().map(|(j, c)| c * design(r, j)).sum();
        let residual = y_r - fitted;
        rss += residual * residual;
    }
    let sd = (rss / (n - p1) as f64).sqrt();
    (sd.is_finite() && sd > 0.0).then_some(sd)
}

fn sample_sd(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let ss = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>();
    (ss / (n - 1.0)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b}");
    }

    #[test]
    fn scaling_maps_training_range_to_half_unit_interval() {
        let x = Data::new(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2).unwrap();
        let y = vec![0.0, 5.0, 10.0];
        let (scaler, xs, ys) = Scaler::fit(&x, &y, &Geometry::euclidean(2));
        assert_eq!(ys, vec![-0.5, 0.0, 0.5]);
        assert_eq!(xs.row(0), &[-0.5, -0.5]);
        assert_eq!(xs.row(2), &[0.5, 0.5]);
        close(scaler.unscale_y(scaler.scale_y(3.3)), 3.3, 1e-12);
        assert_eq!(scaler.y_range(), 10.0);
        let outside = Data::new(vec![4.0, 0.0], 1, 2).unwrap();
        assert_eq!(scaler.scale_x(&outside).row(0), &[1.0, -1.0]);
    }

    #[test]
    fn unscaled_columns_keep_the_caller_scale() {
        use crate::geometry::Metric;
        let x = Data::new(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2).unwrap();
        let g =
            Geometry::structure(&[Metric::Spherical { sphere: 0 }, Metric::Euclidean], 2).unwrap();
        let (scaler, xs, _) = Scaler::fit(&x, &[0.0, 5.0, 10.0], &g);
        assert_eq!(xs.row(0), &[1.0, -0.5]);
        assert_eq!(xs.row(2), &[3.0, 0.5]);
        assert_eq!((scaler.x_min()[0], scaler.x_max()[0]), (-0.5, 0.5));
    }

    #[test]
    fn serde_rejects_bad_ranges() {
        let json = r#"{"y_min":0.0,"y_max":1.0,"x_min":[0.0],"x_max":[0.0]}"#;
        assert!(serde_json::from_str::<Scaler>(json).is_err());
        let json = r#"{"y_min":0.0,"y_max":1.0,"x_min":[0.0],"x_max":[1.0]}"#;
        let scaler: Scaler = serde_json::from_str(json).unwrap();
        assert_eq!(scaler.n_cols(), 1);
    }

    #[test]
    fn sigma_mu_sq_closed_form() {
        close(
            sigma_mu_sq(3.0, 200),
            (0.5 / (3.0 * 200f64.sqrt())).powi(2),
            1e-18,
        );
    }

    #[test]
    fn probit_sigma_mu_sq_closed_form() {
        close(
            probit_sigma_mu_sq(2.0, 50),
            (3.0 / (2.0 * 50f64.sqrt())).powi(2),
            1e-18,
        );
    }

    #[test]
    fn fit_x_leaves_the_response_map_as_the_identity() {
        let x = Data::new(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2).unwrap();
        let (scaler, xs) = Scaler::fit_x(&x, &Geometry::euclidean(2));
        assert_eq!(xs.row(0), &[-0.5, -0.5]);
        assert_eq!(scaler.scale_y(0.37), 0.37);
        assert_eq!(scaler.unscale_y(-2.5), -2.5);
        assert_eq!(scaler.y_range(), 1.0);
    }

    #[test]
    fn variance_cell_prior_matches_the_mean_of_sigma_sq() {
        // E[prod v_j] = (nu' lambda' / (nu' - 2))^m' = nu lambda / (nu - 2).
        let (nu, lambda) = (6.0, 0.03);
        for m_var in [1, 5, 40] {
            let (nu_p, lambda_p) = variance_cell_prior(nu, lambda, m_var);
            let product = (nu_p * lambda_p / (nu_p - 2.0)).powi(m_var as i32);
            close(product, nu * lambda / (nu - 2.0), 1e-12);
        }
        let (nu_p, lambda_p) = variance_cell_prior(nu, lambda, 1);
        close(nu_p, nu, 1e-12);
        close(lambda_p, lambda, 1e-15);
    }

    #[test]
    fn ols_residual_sd_hand_value() {
        // x = 0..3, y = (1.1, 2.9, 5.1, 6.9): slope 1.96, intercept 1.06,
        // residuals (0.04, -0.12, 0.12, -0.04), RSS 0.032 on 2 degrees of
        // freedom.
        let x = Data::new(vec![0.0, 1.0, 2.0, 3.0], 4, 1).unwrap();
        let y = vec![1.1, 2.9, 5.1, 6.9];
        let sd = ols_residual_sd(&x, &y).unwrap();
        close(sd, (0.032_f64 / 2.0).sqrt(), 1e-9);
        assert!(ols_residual_sd(&Data::new(vec![0.0, 1.0], 2, 1).unwrap(), &[1.0, 2.0]).is_none());
        close(
            sample_sd(&[1.0, 2.0, 3.0, 4.0]),
            (5.0_f64 / 3.0).sqrt(),
            1e-15,
        );
    }

    #[test]
    fn lambda_calibration_hand_value() {
        // nu = 6, q = 0.85: the chi^2_6 quantile at 0.15 is 2.6612732.
        close(
            calibrate_lambda(6.0, 0.85, 0.1),
            0.01 * 2.661_273_176_1 / 6.0,
            1e-10,
        );
    }
}
