//! The shared machinery of the censored latent-normal outcomes: a
//! truncation bound per training row on the scaled response scale, and
//! the truncated-normal refresh of the censored rows each sweep. The
//! tobit model derives its bounds from two shared limits, the AFT
//! model from each row's own censoring time and the interval-censored
//! model from each row's own pair of bounds; the refresh is the same
//! draw either way (Robert 1995 rejection for the truncated normal).

use crate::rng::{self, Rng};

/// The truncation of one training row's latent, scaled response scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Bound {
    /// The response is the latent; no refresh.
    Observed,
    /// The latent lies at or below the value (censored below).
    Below(f64),
    /// The latent lies at or above the value (censored above).
    Above(f64),
    /// The latent lies between the two values (interval-censored).
    Between(f64, f64),
}

/// Refresh the latent of each censored row from N(f_i, 1 / w_i)
/// truncated to its bound, w_i the row's precision; an observed row
/// keeps its response. No randomness is consumed for a response with no
/// censored rows, which is what makes an uncensored fit reproduce the
/// Gaussian chain draw for draw.
pub(crate) fn refresh(
    bounds: &[Bound],
    total: &[f64],
    precision: &[f64],
    y: &mut [f64],
    rng: &mut Rng,
) {
    for (((slot, &bound), &f), &w) in y.iter_mut().zip(bounds).zip(total).zip(precision) {
        match bound {
            Bound::Observed => {}
            Bound::Below(limit) => {
                let sd = 1.0 / w.sqrt();
                *slot = f - sd * rng::truncated_standard_normal_above((f - limit) / sd, rng);
            }
            Bound::Above(limit) => {
                let sd = 1.0 / w.sqrt();
                *slot = f + sd * rng::truncated_standard_normal_above((limit - f) / sd, rng);
            }
            Bound::Between(lower, upper) => {
                let sd = 1.0 / w.sqrt();
                *slot = f + sd
                    * rng::truncated_standard_normal_between(
                        (lower - f) / sd,
                        (upper - f) / sd,
                        rng,
                    );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::chain_rng;
    use rand_core::RngCore;

    #[test]
    fn the_refresh_draws_on_the_censored_side_only() {
        let bounds = [
            Bound::Observed,
            Bound::Below(-0.2),
            Bound::Above(0.3),
            Bound::Between(-0.1, 0.2),
            Bound::Observed,
        ];
        let mut y = [0.1, -0.2, 0.3, 0.05, -0.05];
        let mut rng = chain_rng(17);
        refresh(&bounds, &[0.0; 5], &[4.0; 5], &mut y, &mut rng);
        assert_eq!(y[0], 0.1);
        assert!(y[1] <= -0.2);
        assert!(y[2] >= 0.3);
        assert!((-0.1..=0.2).contains(&y[3]));
        assert_eq!(y[4], -0.05);
    }

    #[test]
    fn observed_rows_consume_no_randomness() {
        let bounds = [Bound::Observed; 3];
        let mut y = [0.1, 0.2, 0.3];
        let mut rng = chain_rng(5);
        let mut untouched = chain_rng(5);
        refresh(&bounds, &[0.0; 3], &[1.0; 3], &mut y, &mut rng);
        assert_eq!(rng.next_u64(), untouched.next_u64());
    }
}
