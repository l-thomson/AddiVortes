//! Broken-sampler fixtures: two mispriced acceptance ratios and the tests
//! showing the small SBC configuration rejects each. The fixtures exist
//! only under `cfg(test)`; every shipped path runs with `Breakage::None`.

use crate::moves::{self, Move, Prior};
use crate::tessellation::Tessellation;

/// The defect in force on a test sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Breakage {
    None,
    /// The add-centre acceptance ratio inflated by e^0.8.
    InflatedAddCentre,
    /// The add-dimension selection ratio without the weight folding: the
    /// reverse-bound terms upstream corrected in 0.6.8.
    DroppedReverseBounds,
    /// The per-cell normaliser of the inverse-gamma integrated likelihood
    /// dropped from the add-centre and remove-centre ratios; no effect on
    /// a Gaussian ensemble, whose normaliser is zero.
    DroppedCellNormaliser,
    /// The subset-prior normalisers e_d dropped from the DART weight
    /// update's acceptance ratio, so every proposal is accepted.
    #[cfg(feature = "experimental")]
    DroppedSubsetNormaliser,
}

/// The shift the defect adds to ln alpha for move `m`; `normaliser` is
/// the cell family's per-cell constant.
pub(crate) fn log_alpha_shift(
    breakage: Breakage,
    m: Move,
    current: &Tessellation,
    proposed: &Tessellation,
    prior: &Prior,
    normaliser: f64,
) -> f64 {
    match breakage {
        Breakage::None => 0.0,
        Breakage::InflatedAddCentre => {
            if m == Move::AddCentre {
                0.8
            } else {
                0.0
            }
        }
        Breakage::DroppedReverseBounds => {
            if m == Move::AddDimension {
                -moves::log_selection_ratio(m, current, proposed, prior)
            } else {
                0.0
            }
        }
        Breakage::DroppedCellNormaliser => match m {
            Move::AddCentre => -normaliser,
            Move::RemoveCentre => normaliser,
            _ => 0.0,
        },
        // Acts in the weight update, not in the structural moves.
        #[cfg(feature = "experimental")]
        Breakage::DroppedSubsetNormaliser => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::Breakage;
    use crate::config::Config;
    use crate::data::Data;
    use crate::sampler::Sampler;

    /// splitmix64 with Box-Muller, sharing nothing with the chain RNG.
    struct SimRng(u64);

    impl SimRng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
        }

        fn normal(&mut self) -> f64 {
            let (u1, u2) = (1.0 - self.uniform(), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }

        /// Gamma(shape, 1) for any positive shape: Marsaglia and Tsang
        /// (2000), boosted below 1.
        #[cfg(feature = "experimental")]
        fn gamma(&mut self, shape: f64) -> f64 {
            if shape < 1.0 {
                return self.gamma(shape + 1.0) * self.uniform().powf(1.0 / shape);
            }
            let d = shape - 1.0 / 3.0;
            let c = 1.0 / (9.0 * d).sqrt();
            loop {
                let z = self.normal();
                let v = (1.0 + c * z).powi(3);
                if v <= 0.0 {
                    continue;
                }
                let u = self.uniform();
                if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
                    return d * v;
                }
            }
        }

        fn poisson(&mut self, lambda: f64) -> usize {
            let target = self.uniform();
            let mut pmf = (-lambda).exp();
            let mut cdf = pmf;
            let mut k = 0;
            while cdf < target && k < 200 {
                k += 1;
                pmf *= lambda / k as f64;
                cdf += pmf;
            }
            k
        }
    }

    const N: usize = 40;
    const P: usize = 3;
    const M: usize = 1;
    const LAMBDA_C: f64 = 2.0;
    const OMEGA: f64 = 0.6;
    const SIGMA_C: f64 = 0.8;
    const LAMBDA: f64 = 0.04;
    /// SBC size: 400 simulations, 19 kept draws at thinning 10 after 100
    /// burn-in sweeps; chi-squared uniformity over 20 rank bins per
    /// quantity (Gaussian: cells, dimensions, sigma^2; heteroscedastic
    /// with m' = 1: cells, variance cells, s^2 at the first row), family
    /// alpha 0.01 Bonferroni-split across the three, chi^2_19 quantile
    /// 39.939.
    const SIMS: usize = 400;
    const KEPT: usize = 19;
    const THIN: usize = 10;
    const BURN: usize = 100;
    const CRITICAL: f64 = 39.939;

    fn rows() -> Vec<[f64; 3]> {
        (0..N)
            .map(|i| {
                [
                    i as f64 / (N - 1) as f64 - 0.5,
                    ((i * 13) % N) as f64 / N as f64 - 0.5,
                    ((i * 29) % N) as f64 / N as f64 - 0.5,
                ]
            })
            .collect()
    }

    /// The models the fixtures run under.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Model {
        Gaussian,
        Heteroscedastic,
    }

    fn config(model: Model) -> Config {
        let config = Config::new()
            .with_m(M)
            .with_lambda_c(LAMBDA_C)
            .with_omega(OMEGA)
            .with_sigma_c(SIGMA_C);
        match model {
            Model::Heteroscedastic => config.with_m_var(1),
            Model::Gaussian => config,
        }
    }

    /// One prior tessellation truncated to full occupancy, by rejection:
    /// (dims, centres, mus).
    fn prior_tessellation(rows: &[[f64; 3]], rng: &mut SimRng) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
        let theta = OMEGA / P as f64;
        let sigma_mu = 0.5 / (3.0 * (M as f64).sqrt());
        loop {
            let b = 1 + rng.poisson(LAMBDA_C);
            let mut d = 1;
            for _ in 0..P - 1 {
                if rng.uniform() < theta {
                    d += 1;
                }
            }
            let mut all: Vec<usize> = (0..P).collect();
            for i in 0..d {
                let j = i + (rng.uniform() * (P - i) as f64) as usize;
                all.swap(i, j.min(P - 1));
            }
            let mut dims = all[..d].to_vec();
            dims.sort_unstable();
            let centres: Vec<f64> = (0..b * d).map(|_| SIGMA_C * rng.normal()).collect();
            let mut occupied = vec![false; b];
            for row in rows {
                let mut best = f64::INFINITY;
                let mut cell = 0;
                for (k, centre) in centres.chunks_exact(d).enumerate() {
                    let key: f64 = dims
                        .iter()
                        .zip(centre)
                        .map(|(&dim, c)| (row[dim] - c) * (row[dim] - c))
                        .sum();
                    if key < best {
                        best = key;
                        cell = k;
                    }
                }
                occupied[cell] = true;
            }
            if occupied.iter().all(|&o| o) {
                let mus = (0..b).map(|_| sigma_mu * rng.normal()).collect();
                return (dims, centres, mus);
            }
        }
    }

    fn nearest(row: &[f64; 3], dims: &[usize], centres: &[f64]) -> usize {
        let d = dims.len();
        let mut best = f64::INFINITY;
        let mut cell = 0;
        for (k, centre) in centres.chunks_exact(d).enumerate() {
            let key: f64 = dims
                .iter()
                .zip(centre)
                .map(|(&dim, c)| (row[dim] - c) * (row[dim] - c))
                .sum();
            if key < best {
                best = key;
                cell = k;
            }
        }
        cell
    }

    /// nu lambda / chi^2_nu at nu = 6: the prior of sigma^2, and with m' = 1
    /// of each variance cell.
    fn prior_variance(rng: &mut SimRng) -> f64 {
        let chi_sq = -2.0 * (rng.uniform().ln() + rng.uniform().ln() + rng.uniform().ln());
        6.0 * LAMBDA / chi_sq
    }

    /// SBC rank chi-squared statistics for the model's three quantities
    /// with the given breakage in force on every fit.
    fn sbc_statistics(breakage: Breakage, model: Model) -> [f64; 3] {
        let rows = rows();
        let x = Data::from_rows(&rows).unwrap();
        let config = config(model);
        let heteroscedastic = model == Model::Heteroscedastic;
        let mut ranks = [[0.0_f64; 20]; 3];
        for sim in 0..SIMS {
            let mut rng = SimRng(4200 + sim as u64);
            let ensemble: Vec<_> = (0..M)
                .map(|_| prior_tessellation(&rows, &mut rng))
                .collect();
            let variance = heteroscedastic.then(|| {
                let (dims, centres, mut values) = prior_tessellation(&rows, &mut rng);
                for v in values.iter_mut() {
                    *v = prior_variance(&mut rng);
                }
                (dims, centres, values)
            });
            let sigma_sq = prior_variance(&mut rng);
            let variance_at = |row: &[f64; 3]| match &variance {
                Some((dims, centres, values)) => values[nearest(row, dims, centres)],
                None => sigma_sq,
            };
            let truth = [
                ensemble.iter().map(|(_, _, m)| m.len()).sum::<usize>() as f64,
                match &variance {
                    Some((_, _, values)) => values.len() as f64,
                    None => ensemble.iter().map(|(d, _, _)| d.len()).sum::<usize>() as f64,
                },
                variance_at(&rows[0]),
            ];
            let y: Vec<f64> = rows
                .iter()
                .map(|row| {
                    let f: f64 = ensemble
                        .iter()
                        .map(|(dims, centres, mus)| mus[nearest(row, dims, centres)])
                        .sum();
                    f + variance_at(row).sqrt() * rng.normal()
                })
                .collect();

            let mut sampler = Sampler::pinned_prior(&config, &x, &y, LAMBDA, sim as u64).unwrap();
            sampler.breakage = breakage;
            for _ in 0..BURN {
                sampler.step();
            }
            let mut draws = [[0.0_f64; KEPT]; 3];
            for kept in 0..KEPT {
                for _ in 0..THIN {
                    sampler.step();
                }
                let cells = |ts: &[crate::tessellation::Tessellation]| {
                    ts.iter().map(|t| t.n_cells()).sum::<usize>() as f64
                };
                let state = [
                    cells(sampler.tessellations()),
                    if heteroscedastic {
                        cells(sampler.variance_tessellations())
                    } else {
                        sampler
                            .tessellations()
                            .iter()
                            .map(|t| t.n_dims())
                            .sum::<usize>() as f64
                    },
                    sampler.noise_variances()[0],
                ];
                for (series, value) in draws.iter_mut().zip(state) {
                    series[kept] = value;
                }
            }
            for q in 0..3 {
                let below = draws[q].iter().filter(|v| **v < truth[q]).count();
                let equal = draws[q].iter().filter(|v| **v == truth[q]).count();
                let rank = below + (rng.uniform() * (equal + 1) as f64) as usize;
                ranks[q][rank.min(below + equal)] += 1.0;
            }
        }
        let expected = SIMS as f64 / 20.0;
        ranks.map(|counts| {
            counts
                .iter()
                .map(|c| (c - expected) * (c - expected) / expected)
                .sum()
        })
    }

    #[test]
    fn unbroken_sampler_passes_the_small_sbc() {
        for model in [Model::Gaussian, Model::Heteroscedastic] {
            let statistics = sbc_statistics(Breakage::None, model);
            for (q, statistic) in statistics.iter().enumerate() {
                assert!(
                    *statistic < CRITICAL,
                    "{model:?}, quantity {q}: {statistic}"
                );
            }
        }
    }

    fn assert_rejected(breakage: Breakage, model: Model) {
        let statistics = sbc_statistics(breakage, model);
        let max = statistics.iter().cloned().fold(0.0, f64::max);
        assert!(max > CRITICAL, "{statistics:?}");
    }

    #[test]
    fn inflated_add_centre_is_rejected() {
        assert_rejected(Breakage::InflatedAddCentre, Model::Gaussian);
    }

    #[test]
    fn dropped_reverse_bounds_are_rejected() {
        assert_rejected(Breakage::DroppedReverseBounds, Model::Gaussian);
    }

    #[test]
    fn dropped_cell_normaliser_is_rejected() {
        assert_rejected(Breakage::DroppedCellNormaliser, Model::Heteroscedastic);
    }

    #[test]
    fn dropped_cell_normaliser_shifts_the_centre_moves_only() {
        use crate::moves::{Move, Prior};
        use crate::tessellation::Tessellation;
        let prior = Prior::euclidean(2, 1.0, 2.0, 0.8);
        let t = |b: usize| Tessellation {
            centres: vec![0.0; b],
            dims: vec![0],
            mus: vec![0.0; b],
        };
        let shift = |m: Move, from: usize, to: usize| {
            super::log_alpha_shift(
                Breakage::DroppedCellNormaliser,
                m,
                &t(from),
                &t(to),
                &prior,
                0.7,
            )
        };
        assert_eq!(shift(Move::AddCentre, 2, 3), -0.7);
        assert_eq!(shift(Move::RemoveCentre, 3, 2), 0.7);
        assert_eq!(shift(Move::Change, 2, 2), 0.0);
        assert_eq!(shift(Move::AddDimension, 2, 2), 0.0);
    }

    #[cfg(feature = "experimental")]
    mod dart {
        use super::*;
        use crate::config::Inclusion;

        const A: f64 = 2.0;
        const B: f64 = 2.0;
        const RHO: f64 = 3.0;
        /// At omega 0.6 most states have one dimension, where e_1 = 1
        /// and the dropped normaliser has no effect; the fixture raises
        /// the dimension prior so the defect acts.
        const DART_OMEGA: f64 = 2.0;
        const DART_M: usize = 1;
        /// The s chain moves by an independence proposal, so the ranks
        /// need heavier thinning than the structural quantities.
        const DART_THIN: usize = 45;

        /// theta from the discrete grid, then s ~ Dirichlet(theta / 3).
        fn prior_state(rng: &mut SimRng) -> ([f64; 3], f64) {
            let k = 1000;
            let mut thetas = Vec::with_capacity(k);
            let mut weights = Vec::with_capacity(k);
            for i in 1..=k {
                let lambda = i as f64 / (k + 1) as f64;
                thetas.push(lambda * RHO / (1.0 - lambda));
                weights.push(lambda.powf(A - 1.0) * (1.0 - lambda).powf(B - 1.0));
            }
            let total: f64 = weights.iter().sum();
            let target = rng.uniform() * total;
            let mut cumulative = 0.0;
            let mut theta = thetas[k - 1];
            for (i, &w) in weights.iter().enumerate() {
                cumulative += w;
                if target < cumulative {
                    theta = thetas[i];
                    break;
                }
            }
            let g = [
                rng.gamma(theta / 3.0).max(f64::MIN_POSITIVE),
                rng.gamma(theta / 3.0).max(f64::MIN_POSITIVE),
                rng.gamma(theta / 3.0).max(f64::MIN_POSITIVE),
            ];
            let total = g[0] + g[1] + g[2];
            ([g[0] / total, g[1] / total, g[2] / total], theta)
        }

        /// A dimension subset with P(S | d) proportional to the product
        /// of member weights, enumerated over p = 3.
        fn weighted_dims(s: &[f64; 3], rng: &mut SimRng) -> Vec<usize> {
            let theta = DART_OMEGA / P as f64;
            let mut d = 1;
            for _ in 0..P - 1 {
                if rng.uniform() < theta {
                    d += 1;
                }
            }
            match d {
                1 => {
                    let total = s[0] + s[1] + s[2];
                    let target = rng.uniform() * total;
                    let mut cumulative = 0.0;
                    for (col, &w) in s.iter().enumerate() {
                        cumulative += w;
                        if target < cumulative {
                            return vec![col];
                        }
                    }
                    vec![2]
                }
                2 => {
                    let pairs = [(0, 1), (0, 2), (1, 2)];
                    let weights = pairs.map(|(i, j)| s[i] * s[j]);
                    let total: f64 = weights.iter().sum();
                    let target = rng.uniform() * total;
                    let mut cumulative = 0.0;
                    for ((i, j), w) in pairs.iter().zip(weights) {
                        cumulative += w;
                        if target < cumulative {
                            return vec![*i, *j];
                        }
                    }
                    vec![1, 2]
                }
                _ => vec![0, 1, 2],
            }
        }

        /// One prior tessellation under the weights, truncated to full
        /// occupancy.
        fn prior_tessellation(
            rows: &[[f64; 3]],
            s: &[f64; 3],
            rng: &mut SimRng,
        ) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
            let sigma_mu = 0.5 / (3.0 * (DART_M as f64).sqrt());
            loop {
                let b = 1 + rng.poisson(LAMBDA_C);
                let dims = weighted_dims(s, rng);
                let d = dims.len();
                let centres: Vec<f64> = (0..b * d).map(|_| SIGMA_C * rng.normal()).collect();
                let mut occupied = vec![false; b];
                for row in rows {
                    occupied[nearest(row, &dims, &centres)] = true;
                }
                if occupied.iter().all(|&o| o) {
                    let mus = (0..b).map(|_| sigma_mu * rng.normal()).collect();
                    return (dims, centres, mus);
                }
            }
        }

        /// SBC rank statistics for (s_0, total dims) with the breakage in
        /// force.
        fn sbc_statistics(breakage: Breakage) -> [f64; 2] {
            let rows = rows();
            let x = Data::from_rows(&rows).unwrap();
            let config = config(Model::Gaussian)
                .with_m(DART_M)
                .with_omega(DART_OMEGA)
                .with_inclusion(Inclusion::Dart {
                    a: A,
                    b: B,
                    rho: Some(RHO),
                });
            let mut ranks = [[0.0_f64; 20]; 2];
            for sim in 0..SIMS {
                let mut rng = SimRng(8400 + sim as u64);
                let (s, _) = prior_state(&mut rng);
                let ensemble: Vec<_> = (0..DART_M)
                    .map(|_| prior_tessellation(&rows, &s, &mut rng))
                    .collect();
                let sigma_sq = prior_variance(&mut rng);
                let truth = [
                    s[0],
                    ensemble.iter().map(|(d, _, _)| d.len()).sum::<usize>() as f64,
                ];
                let y: Vec<f64> = rows
                    .iter()
                    .map(|row| {
                        let f: f64 = ensemble
                            .iter()
                            .map(|(dims, centres, mus)| mus[nearest(row, dims, centres)])
                            .sum();
                        f + sigma_sq.sqrt() * rng.normal()
                    })
                    .collect();

                let mut sampler =
                    Sampler::pinned_prior(&config, &x, &y, LAMBDA, sim as u64).unwrap();
                sampler.breakage = breakage;
                for _ in 0..BURN {
                    sampler.step();
                }
                let mut draws = [[0.0_f64; KEPT]; 2];
                for kept in 0..KEPT {
                    for _ in 0..DART_THIN {
                        sampler.step();
                    }
                    let (weights, _) = sampler.inclusion_state().unwrap();
                    let state = [
                        weights[0],
                        sampler
                            .tessellations()
                            .iter()
                            .map(|t| t.n_dims())
                            .sum::<usize>() as f64,
                    ];
                    for (series, value) in draws.iter_mut().zip(state) {
                        series[kept] = value;
                    }
                }
                for q in 0..2 {
                    let below = draws[q].iter().filter(|v| **v < truth[q]).count();
                    let equal = draws[q].iter().filter(|v| **v == truth[q]).count();
                    let rank = below + (rng.uniform() * (equal + 1) as f64) as usize;
                    ranks[q][rank.min(below + equal)] += 1.0;
                }
            }
            let expected = SIMS as f64 / 20.0;
            ranks.map(|counts| {
                counts
                    .iter()
                    .map(|c| (c - expected) * (c - expected) / expected)
                    .sum()
            })
        }

        #[test]
        fn unbroken_dart_passes_the_small_sbc() {
            let statistics = sbc_statistics(Breakage::None);
            for (q, statistic) in statistics.iter().enumerate() {
                assert!(*statistic < CRITICAL, "quantity {q}: {statistic}");
            }
        }

        #[test]
        fn dropped_subset_normaliser_is_rejected() {
            let statistics = sbc_statistics(Breakage::DroppedSubsetNormaliser);
            let max = statistics.iter().cloned().fold(0.0, f64::max);
            assert!(max > CRITICAL, "{statistics:?}");
        }
    }
}
