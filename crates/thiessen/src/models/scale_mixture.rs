//! Shared machinery of the scale-mixture outcome models: the
//! per-observation weight refresh their conditionals run through.
//!
//! A scale-mixture model holds one precision weight w_i per observation,
//! y_i | w_i ~ N(f(x_i), sigma^2 / w_i), and redraws every w_i each sweep
//! from its conditional given the residual r_i = y_i - f(x_i) and the
//! current sigma^2. The kernel does not pass sigma^2 to
//! [`draw_extra`](crate::outcome::OutcomeModel::draw_extra); the standing
//! precisions carry it exactly. The kernel wrote
//! precision_i = w_i / sigma^2 with this model's own weights at the end
//! of the previous sweep, and no update between the two touches sigma^2,
//! so precision_i / w_i recovers 1 / sigma^2 on every row. Under
//! prior-only sampling the precisions are zero, the recovered value is
//! zero, and every likelihood factor of the weight conditional carries
//! it, so the draw reduces to the prior: the kernel-wide convention that
//! zero precision removes the likelihood from every conditional.
//!
//! The recovery holds only while the model's weights are the ones the
//! precisions were written with, so a model must not reset its weights
//! on the response-replacement path (`init` runs again there).

use crate::rng::Rng;

/// The floor of a drawn weight, the smallest positive normal: a weight
/// underflowing to zero would leave the next sweep's sigma^2 recovery
/// 0 / 0.
pub(crate) const WEIGHT_FLOOR: f64 = f64::MIN_POSITIVE;

/// Redraw every weight in place: `conditional` receives the residual
/// y_i - f(x_i) and the likelihood precision 1 / sigma^2 recovered as
/// precision_i / w_i (zero under prior-only sampling, where the
/// conditional is the prior) and returns the new weight, floored at
/// [`WEIGHT_FLOOR`]. One draw per observation, in row order.
pub(crate) fn refresh_weights(
    weights: &mut [f64],
    y: &[f64],
    total: &[f64],
    precision: &[f64],
    rng: &mut Rng,
    mut conditional: impl FnMut(f64, f64, &mut Rng) -> f64,
) {
    for ((w, &p), (&value, &f)) in weights.iter_mut().zip(precision).zip(y.iter().zip(total)) {
        let scale_precision = p / *w;
        *w = conditional(value - f, scale_precision, rng).max(WEIGHT_FLOOR);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::chain_rng;

    /// The refresh hands each conditional the residual and the recovered
    /// 1 / sigma^2, and floors the returned weight.
    #[test]
    fn the_refresh_recovers_the_scale_and_floors_the_weight() {
        let y = [1.0, 2.0, 3.0];
        let total = [0.5, 0.5, 0.5];
        // precision_i = w_i / sigma^2 with sigma^2 = 4.
        let weights_before = [1.0, 2.0, 0.5];
        let precision: Vec<f64> = weights_before.iter().map(|w| w / 4.0).collect();
        let mut weights = weights_before;
        let mut seen = Vec::new();
        let mut rng = chain_rng(1);
        refresh_weights(&mut weights, &y, &total, &precision, &mut rng, |r, s, _| {
            seen.push((r, s));
            0.0
        });
        for (i, &(r, s)) in seen.iter().enumerate() {
            assert!((r - (y[i] - total[i])).abs() < 1e-15);
            assert!((s - 0.25).abs() < 1e-15, "row {i}: {s}");
        }
        assert!(weights.iter().all(|&w| w == WEIGHT_FLOOR));
    }

    /// Zero precisions reach the conditional as a zero scale precision,
    /// the prior-only convention.
    #[test]
    fn zero_precision_reaches_the_conditional_as_zero() {
        let mut weights = [1.0, 1.0];
        let mut rng = chain_rng(2);
        refresh_weights(
            &mut weights,
            &[0.1, -0.2],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &mut rng,
            |_, s, _| {
                assert_eq!(s, 0.0);
                1.0
            },
        );
        assert_eq!(weights, [1.0, 1.0]);
    }
}
