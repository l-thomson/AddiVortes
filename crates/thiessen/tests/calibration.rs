//! Simulation-based calibration (Talts et al. 2018; Modrák et al. 2025,
//! Bayesian Analysis) and the Geweke (2004) joint-distribution test, run
//! under the pinned prior so the prior does not depend on the data. The
//! harness is parametrised by a [`Model`]: its configuration, its prior
//! simulator, its response generator and its list of test quantities.
//!
//! Each test has two sizes. The small configuration runs in `cargo test`
//! and applies chi-squared and Kolmogorov-Smirnov gates in process. The
//! full configuration is `#[ignore]` and runs in the nightly suite; it
//! applies the same gates and writes ranks and samples as CSV under
//! `CALIBRATION_DIR` (default `target/calibration`) for the R evaluation
//! with the SBC package and ECDF difference bands (Säilynoja, Bürkner
//! and Vehtari 2022).

mod common;
use common::TestRng;
use std::f64::consts::PI;
use thiessen::{Config, Data, Metric, Sampler};

/// Quantities of the Gaussian model, in column order: sigma^2, total cells
/// and total active dimensions over the ensemble, f at three fixed
/// training rows, and the mean of the generated response. SBC ranks the
/// theta functions (the first `n_sbc`); the Geweke test compares all.
const GAUSSIAN_QUANTITIES: [&str; 7] = ["sigma_sq", "cells", "dims", "f_a", "f_b", "f_c", "y_mean"];
/// Quantities of the probit model: no sigma^2; f is the latent mean
/// c + f(x); y_mean is the share of ones.
const PROBIT_QUANTITIES: [&str; 6] = ["cells", "dims", "f_a", "f_b", "f_c", "y_mean"];
/// Quantities of the heteroscedastic model: those of the mean ensemble,
/// the total cells over the variance ensemble, and s^2 at the three rows.
const HETEROSCEDASTIC_QUANTITIES: [&str; 10] = [
    "cells", "dims", "f_a", "f_b", "f_c", "vcells", "s2_a", "s2_b", "s2_c", "y_mean",
];
const F_ROWS: [usize; 3] = [10, 25, 40];

/// Significance 0.01 per test family, Bonferroni-split across the
/// quantities. Gaussian: alpha' = 0.01 / 6 for SBC (chi^2_19 42.198,
/// chi^2_99 145.404), 0.01 / 7 for Geweke (chi^2_7 23.440, chi^2_3
/// 15.510). The probit model has fewer quantities and uses the Gaussian
/// values, which are then conservative. Heteroscedastic: 0.01 / 9 for
/// SBC (43.488, 147.655), 0.01 / 10 for Geweke (24.322, 16.266).
struct Gates {
    sbc_chi2_df19: f64,
    sbc_chi2_df99: f64,
    geweke_chi2_df7: f64,
    geweke_chi2_df3: f64,
    geweke_alpha: f64,
}

const GAUSSIAN_GATES: Gates = Gates {
    sbc_chi2_df19: 42.198,
    sbc_chi2_df99: 145.404,
    geweke_chi2_df7: 23.440,
    geweke_chi2_df3: 15.510,
    geweke_alpha: 0.01 / 7.0,
};

const HETEROSCEDASTIC_GATES: Gates = Gates {
    sbc_chi2_df19: 43.488,
    sbc_chi2_df99: 147.655,
    geweke_chi2_df7: 24.322,
    geweke_chi2_df3: 16.266,
    geweke_alpha: 0.01 / 10.0,
};

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Gaussian,
    Probit,
    Heteroscedastic,
}

/// The centre-coordinate law of one column, the test's own: N(mean, sd^2),
/// wrapped to [-pi, pi] for a longitude.
#[derive(Clone, Copy)]
struct Law {
    mean: f64,
    sd: f64,
    wrapped: bool,
}

impl Law {
    fn draw(&self, rng: &mut TestRng) -> f64 {
        let mut v = self.mean + self.sd * rng.normal();
        if self.wrapped {
            while v > PI {
                v -= 2.0 * PI;
            }
            while v < -PI {
                v += 2.0 * PI;
            }
        }
        v
    }
}

/// One model under test: the pinned-prior configuration and the test
/// quantities, the first `n_sbc` of which are functions of theta alone.
/// `spherical` makes the two columns latitude and longitude of one
/// sphere, with `laws` the per-column coordinate laws.
struct Model {
    kind: Kind,
    config: Config,
    lambda: f64,
    x: Data,
    rows: Vec<[f64; 2]>,
    spherical: bool,
    laws: [Law; 2],
    quantities: &'static [&'static str],
    n_sbc: usize,
    gates: Gates,
}

/// One tessellation of a prior draw: active columns, row-major centres,
/// cell values.
type DrawnTessellation = (Vec<usize>, Vec<f64>, Vec<f64>);

/// One prior draw of the ensembles, engine-free: the test's own sampler.
struct PriorDraw {
    tessellations: Vec<DrawnTessellation>,
    sigma_sq: f64,
    /// The variance tessellations of the heteroscedastic model; empty
    /// otherwise.
    variance: Vec<DrawnTessellation>,
}

/// The Gaussian model at the calibration size: n = 50, p = 2, m = 3,
/// lambda_c = 2, omega = 0.8, sigma_c = 0.8, nu = 6, k = 3, lambda 0.04.
fn gaussian_model() -> Model {
    let n = 50;
    let rows: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            [
                i as f64 / (n - 1) as f64 - 0.5,
                ((i * 17) % n) as f64 / n as f64 - 0.5,
            ]
        })
        .collect();
    let x = Data::from_rows(&rows).unwrap();
    let config = Config::new()
        .with_m(3)
        .with_lambda_c(2.0)
        .with_omega(0.8)
        .with_sigma_c(0.8);
    let euclidean = Law {
        mean: 0.0,
        sd: 0.8,
        wrapped: false,
    };
    Model {
        kind: Kind::Gaussian,
        config,
        lambda: 0.04,
        x,
        rows,
        spherical: false,
        laws: [euclidean, euclidean],
        quantities: &GAUSSIAN_QUANTITIES,
        n_sbc: 6,
        gates: GAUSSIAN_GATES,
    }
}

/// The Gaussian model on one sphere: the calibration rows mapped to
/// latitude in [-pi / 2, pi / 2] and longitude in [-pi, pi), the
/// great-circle metric, and the spherical coordinate laws N(mid, sd^2)
/// with sd = range / (2 Phi^-1(0.75)), the longitude wrapped.
fn spherical_model() -> Model {
    let gaussian = gaussian_model();
    let rows: Vec<[f64; 2]> = gaussian
        .rows
        .iter()
        .map(|r| [r[0] * PI, r[1] * 2.0 * PI])
        .collect();
    let law = |col: usize, wrapped: bool| {
        let (lo, hi) = rows
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), r| {
                (lo.min(r[col]), hi.max(r[col]))
            });
        Law {
            mean: 0.5 * (lo + hi),
            sd: 0.5 * (hi - lo) / 0.674_489_750_196_081_7,
            wrapped,
        }
    };
    let laws = [law(0, false), law(1, true)];
    Model {
        config: gaussian.config.with_metric(vec![
            Metric::Spherical { sphere: 0 },
            Metric::Spherical { sphere: 0 },
        ]),
        x: Data::from_rows(&rows).unwrap(),
        rows,
        spherical: true,
        laws,
        ..gaussian
    }
}

/// The probit model at the calibration size: the Gaussian model's rows
/// and structural prior, offset c = -0.2 fixed, k = 3, no sigma^2.
fn probit_model() -> Model {
    let gaussian = gaussian_model();
    Model {
        kind: Kind::Probit,
        config: gaussian
            .config
            .with_model(thiessen::Model::Probit)
            .with_offset(-0.2),
        lambda: 1.0,
        quantities: &PROBIT_QUANTITIES,
        n_sbc: 5,
        ..gaussian
    }
}

/// The heteroscedastic model at the calibration size: the Gaussian
/// model's rows, structural prior and lambda, with m' = 2 variance
/// tessellations (nu' = 2 / (1 - (2 / 3)^(1 / 2)), lambda' = 0.2).
fn heteroscedastic_model() -> Model {
    let gaussian = gaussian_model();
    Model {
        kind: Kind::Heteroscedastic,
        config: gaussian
            .config
            .with_model(thiessen::Model::Heteroscedastic)
            .with_m_var(2),
        quantities: &HETEROSCEDASTIC_QUANTITIES,
        n_sbc: 9,
        gates: HETEROSCEDASTIC_GATES,
        ..gaussian
    }
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
}

impl Model {
    fn sigma_mu(&self) -> f64 {
        let scale = match self.kind {
            Kind::Gaussian | Kind::Heteroscedastic => 0.5,
            Kind::Probit => 3.0,
        };
        scale / (self.config.k * (self.config.m as f64).sqrt())
    }

    fn offset(&self) -> f64 {
        self.config.offset.unwrap_or(0.0)
    }

    /// (nu', lambda') of one variance cell.
    fn variance_cell_prior(&self) -> (f64, f64) {
        let root = 1.0 / self.config.m_var as f64;
        (
            2.0 / (1.0 - (1.0 - 2.0 / self.config.nu).powf(root)),
            self.lambda.powf(root),
        )
    }

    /// One tessellation from the prior truncated to full occupancy, by
    /// rejection: structure first, cell values after acceptance (occupancy
    /// does not involve the values).
    fn prior_tessellation(
        &self,
        rng: &mut TestRng,
        value: &dyn Fn(&mut TestRng) -> f64,
    ) -> DrawnTessellation {
        let p = self.x.n_cols();
        let theta = self.config.omega.unwrap_or(3.0_f64.min(p as f64)) / p as f64;
        loop {
            let b = 1 + rng.poisson(self.config.lambda_c);
            let mut d = 1;
            for _ in 0..p - 1 {
                if rng.uniform() < theta {
                    d += 1;
                }
            }
            let mut dims: Vec<usize> = (0..p).collect();
            for i in 0..d {
                let j = i + (rng.uniform() * (p - i) as f64) as usize;
                dims.swap(i, j.min(p - 1));
            }
            let mut dims: Vec<usize> = dims[..d].to_vec();
            dims.sort_unstable();
            let mut centres = Vec::with_capacity(b * d);
            for _ in 0..b {
                for &dim in &dims {
                    centres.push(self.laws[dim].draw(rng));
                }
            }
            let mut occupied = vec![false; b];
            for row in &self.rows {
                occupied[self.nearest(row, &dims, &centres)] = true;
            }
            if occupied.iter().all(|&o| o) {
                let values: Vec<f64> = (0..b).map(|_| value(rng)).collect();
                return (dims, centres, values);
            }
        }
    }

    /// theta ~ prior: m truncated mean tessellations; under the Gaussian
    /// model sigma^2 from nu lambda / chi^2_nu (nu = 6: the chi-squared is
    /// a sum of three exponentials); under the heteroscedastic model m'
    /// truncated variance tessellations with Inv-Gamma(nu' / 2,
    /// nu' lambda' / 2) cell values; the probit model's latent variance is
    /// 1.
    fn prior_draw(&self, rng: &mut TestRng) -> PriorDraw {
        let sigma_mu = self.sigma_mu();
        let tessellations = (0..self.config.m)
            .map(|_| self.prior_tessellation(rng, &|rng| sigma_mu * rng.normal()))
            .collect();
        let sigma_sq = match self.kind {
            Kind::Gaussian => {
                let chi_sq = -2.0 * (rng.uniform().ln() + rng.uniform().ln() + rng.uniform().ln());
                self.config.nu * self.lambda / chi_sq
            }
            Kind::Probit | Kind::Heteroscedastic => 1.0,
        };
        let variance = match self.kind {
            Kind::Heteroscedastic => {
                let (nu, lambda) = self.variance_cell_prior();
                (0..self.config.m_var)
                    .map(|_| {
                        self.prior_tessellation(rng, &|rng| 0.5 * nu * lambda / rng.gamma(0.5 * nu))
                    })
                    .collect()
            }
            Kind::Gaussian | Kind::Probit => Vec::new(),
        };
        PriorDraw {
            tessellations,
            sigma_sq,
            variance,
        }
    }

    /// The mean ensemble value at training row `i`.
    fn f_at(&self, draw: &PriorDraw, i: usize) -> f64 {
        draw.tessellations
            .iter()
            .map(|(dims, centres, mus)| mus[self.nearest(&self.rows[i], dims, centres)])
            .sum()
    }

    /// The variance of y_i given f at training row `i`.
    fn variance_at(&self, draw: &PriorDraw, i: usize) -> f64 {
        match self.kind {
            Kind::Heteroscedastic => draw
                .variance
                .iter()
                .map(|(dims, centres, values)| values[self.nearest(&self.rows[i], dims, centres)])
                .product(),
            Kind::Gaussian | Kind::Probit => draw.sigma_sq,
        }
    }

    /// Squared distance from `row` to a centre over `dims`: Euclidean, or
    /// the great-circle angle with the centre placed at the row's own
    /// coordinate in an inactive column.
    fn key(&self, row: &[f64; 2], dims: &[usize], centre: &[f64]) -> f64 {
        if !self.spherical {
            return dims
                .iter()
                .zip(centre)
                .map(|(&dim, c)| (row[dim] - c) * (row[dim] - c))
                .sum();
        }
        let coordinate = |col: usize| match dims.iter().position(|&dim| dim == col) {
            Some(j) => centre[j],
            None => row[col],
        };
        let (lat, lon) = (coordinate(0), coordinate(1));
        let cos_angle = (row[0].sin() * lat.sin()
            + row[0].cos() * lat.cos() * (row[1] - lon).cos())
        .clamp(-1.0, 1.0);
        cos_angle.acos().powi(2)
    }

    /// Nearest centre of `centres` (row-major, one coordinate per active
    /// column) to `row`; ties to the lowest index, matching the engine.
    fn nearest(&self, row: &[f64; 2], dims: &[usize], centres: &[f64]) -> usize {
        let mut best = f64::INFINITY;
        let mut cell = 0;
        for (k, centre) in centres.chunks_exact(dims.len()).enumerate() {
            let key = self.key(row, dims, centre);
            if key < best {
                best = key;
                cell = k;
            }
        }
        cell
    }

    /// y | theta: f + s e, or labels Bernoulli(Phi(c + f)).
    fn generate_y(&self, draw: &PriorDraw, rng: &mut TestRng) -> Vec<f64> {
        (0..self.x.n_rows())
            .map(|i| {
                let f = self.f_at(draw, i);
                match self.kind {
                    Kind::Gaussian | Kind::Heteroscedastic => {
                        f + self.variance_at(draw, i).sqrt() * rng.normal()
                    }
                    Kind::Probit => f64::from(rng.uniform() < normal_cdf(self.offset() + f)),
                }
            })
            .collect()
    }

    /// The quantities of a marginal-conditional draw.
    fn mc_quantities(&self, draw: &PriorDraw, y: &[f64]) -> Vec<f64> {
        let cells: usize = draw.tessellations.iter().map(|(_, _, m)| m.len()).sum();
        let dims: usize = draw.tessellations.iter().map(|(d, _, _)| d.len()).sum();
        let mut out = Vec::with_capacity(self.quantities.len());
        if self.kind == Kind::Gaussian {
            out.push(draw.sigma_sq);
        }
        out.push(cells as f64);
        out.push(dims as f64);
        out.extend(F_ROWS.iter().map(|&r| self.offset() + self.f_at(draw, r)));
        if self.kind == Kind::Heteroscedastic {
            let vcells: usize = draw.variance.iter().map(|(_, _, v)| v.len()).sum();
            out.push(vcells as f64);
            out.extend(F_ROWS.iter().map(|&r| self.variance_at(draw, r)));
        }
        out.push(y.iter().sum::<f64>() / y.len() as f64);
        out
    }

    /// The quantities of the current sampler state and the response it
    /// conditioned on.
    fn sampler_quantities(&self, sampler: &Sampler, y: &[f64]) -> Vec<f64> {
        let cells: usize = sampler.tessellations().iter().map(|t| t.n_cells()).sum();
        let dims: usize = sampler.tessellations().iter().map(|t| t.n_dims()).sum();
        let fit = sampler.fitted_values();
        let mut out = Vec::with_capacity(self.quantities.len());
        if self.kind == Kind::Gaussian {
            out.push(sampler.sigma_sq());
        }
        out.push(cells as f64);
        out.push(dims as f64);
        out.extend(F_ROWS.iter().map(|&r| fit[r]));
        if self.kind == Kind::Heteroscedastic {
            let vcells: usize = sampler
                .variance_tessellations()
                .iter()
                .map(|t| t.n_cells())
                .sum();
            let variances = sampler.noise_variances();
            out.push(vcells as f64);
            out.extend(F_ROWS.iter().map(|&r| variances[r]));
        }
        out.push(y.iter().sum::<f64>() / y.len() as f64);
        out
    }

    /// One y | theta transition of the successive-conditional chain,
    /// from the sampler's current state.
    fn regenerate_y(&self, sampler: &Sampler, y: &mut [f64], rng: &mut TestRng) {
        let fit = sampler.fitted_values();
        let variances = sampler.noise_variances();
        for ((slot, f), v) in y.iter_mut().zip(&fit).zip(&variances) {
            *slot = match self.kind {
                Kind::Gaussian | Kind::Heteroscedastic => f + v.sqrt() * rng.normal(),
                Kind::Probit => f64::from(rng.uniform() < normal_cdf(*f)),
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Simulation-based calibration
// ---------------------------------------------------------------------------

/// Ranks of the first `n_sbc` quantities over `sims` simulations: theta from
/// the prior, y from the model, `kept` posterior draws at `thinning`
/// after `burn_in`, rank of the theta quantity among the draws with
/// uniform tie-breaking (Talts et al. 2018, s. 4).
fn sbc_ranks(
    model: &Model,
    sims: usize,
    kept: usize,
    thinning: usize,
    burn_in: usize,
    seed: u64,
) -> Vec<Vec<usize>> {
    let mut ranks: Vec<Vec<usize>> = (0..model.n_sbc).map(|_| Vec::with_capacity(sims)).collect();
    for sim in 0..sims {
        let mut rng = TestRng(seed ^ (sim as u64).wrapping_mul(0x9E37_79B9));
        // A constant response is rejected at the fit boundary. Redrawing
        // (theta, y) until y varies selects on y alone, and the ranks stay
        // uniform conditional on y.
        let (draw, y) = loop {
            let draw = model.prior_draw(&mut rng);
            let y = model.generate_y(&draw, &mut rng);
            if y.iter().any(|&v| v != y[0]) {
                break (draw, y);
            }
        };
        let truth = model.mc_quantities(&draw, &y);
        let mut sampler =
            Sampler::pinned_prior(&model.config, &model.x, &y, model.lambda, seed + sim as u64)
                .unwrap();
        for _ in 0..burn_in {
            sampler.step();
        }
        let mut draws: Vec<Vec<f64>> = (0..model.n_sbc).map(|_| Vec::with_capacity(kept)).collect();
        for _ in 0..kept {
            for _ in 0..thinning {
                sampler.step();
            }
            let state = model.sampler_quantities(&sampler, &y);
            for (q, series) in draws.iter_mut().enumerate() {
                series.push(state[q]);
            }
        }
        for (q, series) in draws.iter().enumerate() {
            let below = series.iter().filter(|v| **v < truth[q]).count();
            let equal = series.iter().filter(|v| **v == truth[q]).count();
            let rank = below + (rng.uniform() * (equal + 1) as f64) as usize;
            ranks[q].push(rank.min(below + equal));
        }
    }
    ranks
}

/// Chi-squared uniformity statistic of ranks on 0..=kept.
fn rank_uniformity(ranks: &[usize], kept: usize) -> f64 {
    let bins = kept + 1;
    let mut counts = vec![0.0_f64; bins];
    for &r in ranks {
        counts[r] += 1.0;
    }
    let expected = ranks.len() as f64 / bins as f64;
    counts
        .iter()
        .map(|c| (c - expected) * (c - expected) / expected)
        .sum()
}

fn assert_uniform(model: &Model, ranks: &[Vec<usize>], kept: usize) {
    let critical = if kept == 19 {
        model.gates.sbc_chi2_df19
    } else {
        model.gates.sbc_chi2_df99
    };
    for (q, series) in ranks.iter().enumerate() {
        let statistic = rank_uniformity(series, kept);
        assert!(
            statistic < critical,
            "{}: chi-squared {statistic} over {} bins, critical {critical}",
            model.quantities[q],
            kept + 1
        );
    }
}

#[test]
fn sbc_small_ranks_are_uniform() {
    let model = gaussian_model();
    let ranks = sbc_ranks(&model, 160, 19, 15, 150, 401);
    assert_uniform(&model, &ranks, 19);
}

#[test]
fn sbc_small_ranks_are_uniform_probit() {
    let model = probit_model();
    let ranks = sbc_ranks(&model, 160, 19, 15, 150, 402);
    assert_uniform(&model, &ranks, 19);
}

#[test]
fn sbc_small_ranks_are_uniform_heteroscedastic() {
    let model = heteroscedastic_model();
    let ranks = sbc_ranks(&model, 160, 19, 15, 150, 403);
    assert_uniform(&model, &ranks, 19);
}

#[test]
fn sbc_small_ranks_are_uniform_spherical() {
    let model = spherical_model();
    let ranks = sbc_ranks(&model, 160, 19, 15, 150, 404);
    assert_uniform(&model, &ranks, 19);
}

/// Files first, so a failed gate still leaves the R evaluation its input.
fn sbc_full(model: &Model, file: &str, seed: u64) {
    let ranks = sbc_ranks(model, 1000, 99, 20, 300, seed);
    let mut lines = vec!["quantity,rank,max_rank".to_string()];
    for (q, series) in ranks.iter().enumerate() {
        for rank in series {
            lines.push(format!("{},{rank},99", model.quantities[q]));
        }
    }
    write_csv(file, &lines);
    assert_uniform(model, &ranks, 99);
}

#[test]
#[ignore = "full size, nightly"]
fn sbc_full_ranks_are_uniform() {
    sbc_full(&gaussian_model(), "sbc_ranks.csv", 401);
}

#[test]
#[ignore = "full size, nightly"]
fn sbc_full_ranks_are_uniform_probit() {
    sbc_full(&probit_model(), "sbc_ranks_probit.csv", 402);
}

#[test]
#[ignore = "full size, nightly"]
fn sbc_full_ranks_are_uniform_heteroscedastic() {
    sbc_full(
        &heteroscedastic_model(),
        "sbc_ranks_heteroscedastic.csv",
        403,
    );
}

#[test]
#[ignore = "full size, nightly"]
fn sbc_full_ranks_are_uniform_spherical() {
    sbc_full(&spherical_model(), "sbc_ranks_spherical.csv", 404);
}

// ---------------------------------------------------------------------------
// Geweke joint-distribution test
// ---------------------------------------------------------------------------

/// Marginal-conditional against successive-conditional samples per
/// quantity (Geweke 2004). The successive-conditional chain alternates
/// y | theta (through `set_response`) with theta | y (one sweep of the
/// kernel under test), discards `discard` transitions and keeps every
/// `thin`-th of the next `n_sc * thin`. The Kolmogorov-Smirnov critical
/// value assumes independent draws; thinning 45 leaves the lag-one
/// autocorrelation of every quantity below 0.1 for each model.
fn geweke_samples(
    model: &Model,
    n_mc: usize,
    n_sc: usize,
    thin: usize,
    discard: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut rng = TestRng(seed);
    let n_q = model.quantities.len();
    let mut mc: Vec<Vec<f64>> = (0..n_q).map(|_| Vec::with_capacity(n_mc)).collect();
    for _ in 0..n_mc {
        let draw = model.prior_draw(&mut rng);
        let y = model.generate_y(&draw, &mut rng);
        let state = model.mc_quantities(&draw, &y);
        for (q, series) in mc.iter_mut().enumerate() {
            series.push(state[q]);
        }
    }

    let first = model.prior_draw(&mut rng);
    let mut y = model.generate_y(&first, &mut rng);
    let mut sampler =
        Sampler::pinned_prior(&model.config, &model.x, &y, model.lambda, seed).unwrap();
    let mut sc: Vec<Vec<f64>> = (0..n_q).map(|_| Vec::with_capacity(n_sc)).collect();
    let transition = |sampler: &mut Sampler, y: &mut Vec<f64>, rng: &mut TestRng| {
        model.regenerate_y(sampler, y, rng);
        sampler.set_response(y).unwrap();
        sampler.step();
    };
    for _ in 0..discard {
        transition(&mut sampler, &mut y, &mut rng);
    }
    for _ in 0..n_sc {
        for _ in 0..thin {
            transition(&mut sampler, &mut y, &mut rng);
        }
        let state = model.sampler_quantities(&sampler, &y);
        for (q, series) in sc.iter_mut().enumerate() {
            series.push(state[q]);
        }
    }
    (mc, sc)
}

/// Two-sample Kolmogorov-Smirnov statistic.
fn ks_statistic(a: &[f64], b: &[f64]) -> f64 {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    a.sort_unstable_by(f64::total_cmp);
    b.sort_unstable_by(f64::total_cmp);
    let (mut i, mut j, mut d) = (0_usize, 0_usize, 0.0_f64);
    while i < a.len() && j < b.len() {
        let t = a[i].min(b[j]);
        while i < a.len() && a[i] <= t {
            i += 1;
        }
        while j < b.len() && b[j] <= t {
            j += 1;
        }
        let gap = (i as f64 / a.len() as f64 - j as f64 / b.len() as f64).abs();
        if gap > d {
            d = gap;
        }
    }
    d
}

/// Two-sample chi-squared statistic over fixed bins.
fn chi_squared_binned(a: &[f64], b: &[f64], lo: usize, hi: usize) -> f64 {
    let bins = hi - lo + 1;
    let index = |v: f64| ((v as usize).clamp(lo, hi)) - lo;
    let mut counts = vec![[0.0_f64; 2]; bins];
    for &v in a {
        counts[index(v)][0] += 1.0;
    }
    for &v in b {
        counts[index(v)][1] += 1.0;
    }
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    let (r1, r2) = ((n2 / n1).sqrt(), (n1 / n2).sqrt());
    counts
        .iter()
        .filter(|c| c[0] + c[1] > 0.0)
        .map(|c| {
            let diff = r1 * c[0] - r2 * c[1];
            diff * diff / (c[0] + c[1])
        })
        .sum()
}

fn assert_simulators_agree(model: &Model, mc: &[Vec<f64>], sc: &[Vec<f64>]) {
    let gates = &model.gates;
    for (q, name) in model.quantities.iter().enumerate() {
        match *name {
            // Total cells over m = 3 tessellations, bins <=5 to >=12;
            // total dimensions, bins 3 to 6; total variance cells over
            // m' = 2 tessellations, bins <=2 to >=9.
            "cells" => {
                let statistic = chi_squared_binned(&mc[q], &sc[q], 5, 12);
                assert!(
                    statistic < gates.geweke_chi2_df7,
                    "cells: chi-squared {statistic}, critical {}",
                    gates.geweke_chi2_df7
                );
            }
            "dims" => {
                let statistic = chi_squared_binned(&mc[q], &sc[q], 3, 6);
                assert!(
                    statistic < gates.geweke_chi2_df3,
                    "dims: chi-squared {statistic}, critical {}",
                    gates.geweke_chi2_df3
                );
            }
            "vcells" => {
                let statistic = chi_squared_binned(&mc[q], &sc[q], 2, 9);
                assert!(
                    statistic < gates.geweke_chi2_df7,
                    "vcells: chi-squared {statistic}, critical {}",
                    gates.geweke_chi2_df7
                );
            }
            _ => {
                let d = ks_statistic(&mc[q], &sc[q]);
                let c = (-(gates.geweke_alpha / 2.0).ln() / 2.0).sqrt();
                let critical = c
                    * ((mc[q].len() + sc[q].len()) as f64 / (mc[q].len() * sc[q].len()) as f64)
                        .sqrt();
                assert!(d < critical, "{name}: KS {d}, critical {critical}");
            }
        }
    }
}

#[test]
fn geweke_small_simulators_agree() {
    let model = gaussian_model();
    let (mc, sc) = geweke_samples(&model, 2000, 800, 45, 200, 907);
    assert_simulators_agree(&model, &mc, &sc);
}

#[test]
fn geweke_small_simulators_agree_probit() {
    let model = probit_model();
    let (mc, sc) = geweke_samples(&model, 2000, 800, 45, 200, 908);
    assert_simulators_agree(&model, &mc, &sc);
}

#[test]
fn geweke_small_simulators_agree_heteroscedastic() {
    let model = heteroscedastic_model();
    let (mc, sc) = geweke_samples(&model, 2000, 800, 45, 200, 909);
    assert_simulators_agree(&model, &mc, &sc);
}

#[test]
fn geweke_small_simulators_agree_spherical() {
    let model = spherical_model();
    let (mc, sc) = geweke_samples(&model, 2000, 800, 45, 200, 910);
    assert_simulators_agree(&model, &mc, &sc);
}

/// Files first, so a failed gate still leaves the R evaluation its input.
fn geweke_full(model: &Model, file: &str, seed: u64) {
    let (mc, sc) = geweke_samples(model, 20_000, 5000, 45, 500, seed);
    let mut lines = vec!["quantity,simulator,value".to_string()];
    for (q, name) in model.quantities.iter().enumerate() {
        for v in &mc[q] {
            lines.push(format!("{name},mc,{v}"));
        }
        for v in &sc[q] {
            lines.push(format!("{name},sc,{v}"));
        }
    }
    write_csv(file, &lines);
    assert_simulators_agree(model, &mc, &sc);
}

#[test]
#[ignore = "full size, nightly"]
fn geweke_full_simulators_agree() {
    geweke_full(&gaussian_model(), "geweke_samples.csv", 907);
}

#[test]
#[ignore = "full size, nightly"]
fn geweke_full_simulators_agree_probit() {
    geweke_full(&probit_model(), "geweke_samples_probit.csv", 908);
}

#[test]
#[ignore = "full size, nightly"]
fn geweke_full_simulators_agree_heteroscedastic() {
    geweke_full(
        &heteroscedastic_model(),
        "geweke_samples_heteroscedastic.csv",
        909,
    );
}

#[test]
#[ignore = "full size, nightly"]
fn geweke_full_simulators_agree_spherical() {
    geweke_full(&spherical_model(), "geweke_samples_spherical.csv", 910);
}

fn write_csv(name: &str, lines: &[String]) {
    // The test binary's working directory is the crate, two levels below
    // the workspace target directory.
    let dir = std::env::var("CALIBRATION_DIR").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/calibration").into()
    });
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(format!("{dir}/{name}"), lines.join("\n") + "\n").unwrap();
}
