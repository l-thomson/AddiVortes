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
    }
}

#[cfg(test)]
mod tests {
    use super::Breakage;
    use crate::config::Config;
    use crate::data::Data;
    use crate::model::Model;
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

    fn config(model: Model) -> Config {
        Config::new()
            .with_model(model)
            .with_m(M)
            .with_m_var(1)
            .with_lambda_c(LAMBDA_C)
            .with_omega(OMEGA)
            .with_sigma_c(SIGMA_C)
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
                assert!(*statistic < CRITICAL, "{model}, quantity {q}: {statistic}");
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
        let prior = Prior {
            p: 2,
            omega: 1.0,
            lambda_c: 2.0,
            sigma_c: 0.8,
        };
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
}
