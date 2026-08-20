//! The chain's random number generator: ChaCha8 keyed from a `u64` seed
//! through splitmix64, and the draws the sampler makes from it.

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use rand_distr::Distribution;

/// The chain RNG.
pub(crate) type Rng = ChaCha8Rng;

/// splitmix64 (Steele, Lea and Flood 2014), one output per call.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The chain RNG for `seed`: the 256-bit ChaCha8 key is four successive
/// splitmix64 outputs from `seed`.
pub(crate) fn chain_rng(seed: u64) -> Rng {
    let mut state = seed;
    let mut key = [0u8; 32];
    for word in key.chunks_exact_mut(8) {
        word.copy_from_slice(&splitmix64(&mut state).to_le_bytes());
    }
    ChaCha8Rng::from_seed(key)
}

/// The seed of chain `index` (0-based) derived from `seed`: chain 0 is
/// `seed` itself, chain k the k-th splitmix64 output after it.
pub fn chain_seed(seed: u64, index: usize) -> u64 {
    let mut state = seed;
    let mut out = seed;
    for _ in 0..index {
        out = splitmix64(&mut state);
    }
    out
}

/// Uniform on [0, 1) with 53 random bits.
pub(crate) fn uniform(rng: &mut Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Uniform index in `0..n`, n >= 1.
pub(crate) fn uniform_index(n: usize, rng: &mut Rng) -> usize {
    debug_assert!(n >= 1);
    ((uniform(rng) * n as f64) as usize).min(n - 1)
}

/// Standard normal draw.
pub(crate) fn standard_normal(rng: &mut Rng) -> f64 {
    rand_distr::StandardNormal.sample(rng)
}

/// Gamma(shape, scale) draw, shape > 0, scale > 0.
pub(crate) fn gamma(shape: f64, scale: f64, rng: &mut Rng) -> f64 {
    rand_distr::Gamma::new(shape, scale)
        .expect("shape and scale are positive by construction")
        .sample(rng)
}

/// Standard normal draw restricted to z >= a (Robert 1995, Statistics and
/// Computing 5, 121-125): plain rejection from N(0, 1) for a < 0.45,
/// otherwise rejection from the exponential with rate
/// (a + sqrt(a^2 + 4)) / 2 shifted to a, whose acceptance rate does not
/// fall as a grows.
pub(crate) fn truncated_standard_normal_above(a: f64, rng: &mut Rng) -> f64 {
    if a < 0.45 {
        loop {
            let z = standard_normal(rng);
            if z >= a {
                return z;
            }
        }
    }
    let alpha = 0.5 * (a + (a * a + 4.0).sqrt());
    loop {
        // 1 - u lies in (0, 1], so the logarithm is finite.
        let z = a - libm::log(1.0 - uniform(rng)) / alpha;
        let rho = libm::exp(-0.5 * (z - alpha) * (z - alpha));
        if uniform(rng) <= rho {
            return z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_normal_moments_on_both_branches() {
        // E[Z | Z >= a] = phi(a) / (1 - Phi(a)).
        let mean_above = |a: f64| {
            let phi = (-0.5 * a * a).exp() / (2.0 * std::f64::consts::PI).sqrt();
            phi / (0.5 * libm::erfc(a * std::f64::consts::FRAC_1_SQRT_2))
        };
        let mut rng = chain_rng(9);
        let n = 200_000;
        for a in [-1.5, 0.0, 0.44, 0.46, 2.0, 6.0] {
            let mut sum = 0.0;
            for _ in 0..n {
                let z = truncated_standard_normal_above(a, &mut rng);
                assert!(z >= a);
                sum += z;
            }
            let mean = sum / n as f64;
            assert!(
                (mean - mean_above(a)).abs() < 0.01,
                "a = {a}: {mean} vs {}",
                mean_above(a)
            );
        }
    }

    #[test]
    fn same_seed_same_stream() {
        let mut a = chain_rng(42);
        let mut b = chain_rng(42);
        let mut c = chain_rng(43);
        let (ua, ub, uc) = (uniform(&mut a), uniform(&mut b), uniform(&mut c));
        assert_eq!(ua, ub);
        assert_ne!(ua, uc);
    }

    #[test]
    fn chain_seeds_are_distinct_and_start_at_the_seed() {
        assert_eq!(chain_seed(7, 0), 7);
        let seeds: Vec<u64> = (0..4).map(|k| chain_seed(7, k)).collect();
        for i in 0..4 {
            for j in 0..i {
                assert_ne!(seeds[i], seeds[j]);
            }
        }
    }

    #[test]
    fn uniform_index_is_in_range() {
        let mut rng = chain_rng(1);
        for _ in 0..1000 {
            assert!(uniform_index(3, &mut rng) < 3);
            assert_eq!(uniform_index(1, &mut rng), 0);
        }
    }

    #[test]
    fn gamma_and_normal_moments() {
        let mut rng = chain_rng(5);
        let n = 20_000;
        let mean_g: f64 = (0..n).map(|_| gamma(4.0, 0.5, &mut rng)).sum::<f64>() / n as f64;
        assert!((mean_g - 2.0).abs() < 0.05, "{mean_g}");
        let mean_z: f64 = (0..n).map(|_| standard_normal(&mut rng)).sum::<f64>() / n as f64;
        assert!(mean_z.abs() < 0.03, "{mean_z}");
    }
}
