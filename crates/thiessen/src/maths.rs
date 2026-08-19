//! Transcendental functions through `libm`, so the reference target does not
//! depend on the system libc, and the special functions the prior
//! calibration and the predictive distribution use.

pub(crate) fn ln(x: f64) -> f64 {
    libm::log(x)
}

pub(crate) fn exp(x: f64) -> f64 {
    libm::exp(x)
}

pub(crate) fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

pub(crate) fn lgamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// Standard normal CDF, Phi(z) = erfc(-z / sqrt 2) / 2.
pub(crate) fn normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
}

/// Regularised lower incomplete gamma function P(a, x), a > 0, x >= 0:
/// series for x < a + 1, continued fraction otherwise (Press et al. 2007,
/// s. 6.2).
pub(crate) fn gamma_p(a: f64, x: f64) -> f64 {
    debug_assert!(a > 0.0 && x >= 0.0);
    if x == 0.0 {
        return 0.0;
    }
    let log_prefactor = a * ln(x) - x - lgamma(a);
    if x < a + 1.0 {
        let mut term = 1.0 / a;
        let mut sum = term;
        let mut denominator = a;
        for _ in 0..500 {
            denominator += 1.0;
            term *= x / denominator;
            sum += term;
            if term.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        sum * exp(log_prefactor)
    } else {
        const TINY: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..500 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        1.0 - exp(log_prefactor) * h
    }
}

/// Quantile of chi^2_nu at probability p in (0, 1), by bisection on
/// [`gamma_p`].
pub(crate) fn chi2_quantile(p: f64, nu: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0 && nu > 0.0);
    let a = 0.5 * nu;
    let mut hi = nu.max(1.0);
    while gamma_p(a, 0.5 * hi) < p {
        hi *= 2.0;
    }
    let mut lo = 0.0_f64;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if gamma_p(a, 0.5 * mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Type 7 quantile of an ascending-sorted non-empty slice: h = p (n - 1),
/// linear interpolation between neighbours.
pub(crate) fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = p * (n - 1) as f64;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (h - lo as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b}");
    }

    #[test]
    fn normal_cdf_reference_values() {
        close(normal_cdf(0.0), 0.5, 1e-15);
        close(normal_cdf(1.959_963_984_540_054), 0.975, 1e-12);
        close(normal_cdf(-3.0), 0.001_349_898_031_630_09, 1e-12);
    }

    #[test]
    fn gamma_p_identities() {
        close(gamma_p(1.0, 2.0), 1.0 - exp(-2.0), 1e-14);
        close(gamma_p(0.5, 0.5), 0.682_689_492_137_086, 1e-12);
        close(gamma_p(5.0, 20.0), 0.999_983_055_256, 1e-10);
    }

    #[test]
    fn chi2_quantile_reference_values() {
        close(chi2_quantile(0.95, 1.0), 3.841_458_820_694_12, 1e-9);
        close(chi2_quantile(0.15, 6.0), 2.661_273_176_1, 1e-8);
        close(chi2_quantile(0.5, 10.0), 9.341_817_765_6, 1e-8);
    }

    #[test]
    fn quantile_sorted_interpolates() {
        let v = [1.0, 2.0, 3.0, 4.0];
        close(quantile_sorted(&v, 0.5), 2.5, 0.0);
        close(quantile_sorted(&v, 0.0), 1.0, 0.0);
        close(quantile_sorted(&v, 1.0), 4.0, 0.0);
        close(quantile_sorted(&[7.0], 0.3), 7.0, 0.0);
    }
}
