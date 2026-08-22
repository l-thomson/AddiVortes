//! Simulation-based calibration (Talts et al. 2018; Modrák et al. 2025,
//! Bayesian Analysis) and the Geweke (2004) joint-distribution test, run
//! under the pinned prior so the prior does not depend on the data. The
//! harness is parametrised by a model under test: its configuration, its
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
#[cfg(feature = "experimental")]
use thiessen::{Basis, GowerKind, Inclusion};
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

#[cfg(feature = "experimental")]
const DART_QUANTITIES: [&str; 9] = [
    "sigma_sq", "cells", "dims", "f_a", "f_b", "f_c", "s0", "theta", "y_mean",
];

/// The Gaussian outcome's sigma^2 prior (nu, q) of a configuration.
fn gaussian_prior(config: &Config) -> (f64, f64) {
    match &config.outcome {
        thiessen::Outcome::Gaussian(params) => (params.nu, params.q),
        _ => unreachable!("the tests read nu and q under gaussian"),
    }
}

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
/// wrapped to [-pi, pi] for a longitude; uniform over the levels of a
/// categorical column.
#[derive(Clone)]
enum Law {
    Normal { mean: f64, sd: f64, wrapped: bool },
    Uniform { levels: Vec<f64> },
}

impl Law {
    fn draw(&self, rng: &mut TestRng) -> f64 {
        match self {
            Law::Normal { mean, sd, wrapped } => {
                let mut v = mean + sd * rng.normal();
                if *wrapped {
                    while v > PI {
                        v -= 2.0 * PI;
                    }
                    while v < -PI {
                        v += 2.0 * PI;
                    }
                }
                v
            }
            Law::Uniform { levels } => levels[(rng.uniform() * levels.len() as f64) as usize],
        }
    }
}

/// The column structure of a model under test: two Euclidean columns; one
/// sphere of latitude and longitude; or a Euclidean column and a
/// categorical column whose mismatch weighs `weight`.
#[derive(Clone, Copy, PartialEq)]
enum Space {
    Euclidean,
    Sphere,
    Categorical {
        weight: f64,
    },
    #[cfg(feature = "experimental")]
    Minkowski {
        p: f64,
    },
    #[cfg(feature = "experimental")]
    Cosine,
    #[cfg(feature = "experimental")]
    Gower,
    #[cfg(feature = "experimental")]
    Mahalanobis {
        precision: [f64; 4],
    },
    #[cfg(feature = "experimental")]
    Composite {
        weight: f64,
    },
}

/// One model under test: the pinned-prior configuration and the test
/// quantities, the first `n_sbc` of which are functions of theta alone;
/// `space` and `laws` the column structure and per-column coordinate
/// laws.
struct Model {
    kind: Kind,
    config: Config,
    lambda: f64,
    x: Data,
    rows: Vec<[f64; 2]>,
    space: Space,
    laws: [Law; 2],
    /// The inclusion weight per column; None is the uniform prior.
    weights: Option<[f64; 2]>,
    /// The DART hyperparameters (a, b, rho); None is a fixed prior.
    dart: Option<(f64, f64, f64)>,
    /// The linear cell basis on the mean ensemble.
    linear: bool,
    quantities: &'static [&'static str],
    n_sbc: usize,
    gates: Gates,
}

/// One tessellation of a prior draw: active columns, row-major centres,
/// cell values, row-major cell slopes (empty under the constant basis).
type DrawnTessellation = (Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>);

/// One prior draw of the ensembles, engine-free: the test's own sampler.
struct PriorDraw {
    tessellations: Vec<DrawnTessellation>,
    sigma_sq: f64,
    /// The variance tessellations of the heteroscedastic model; empty
    /// otherwise.
    variance: Vec<DrawnTessellation>,
    /// The DART weights and concentration; None under a fixed prior.
    dart: Option<([f64; 2], f64)>,
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
    let euclidean = Law::Normal {
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
        space: Space::Euclidean,
        laws: [euclidean.clone(), euclidean],
        weights: None,
        dart: None,
        linear: false,
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
        Law::Normal {
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
        space: Space::Sphere,
        laws,
        ..gaussian
    }
}

/// The Gaussian model with a categorical second column: four levels
/// (codes 0 to 3), mismatch weight 2 / 16, uniform coordinate law.
fn categorical_model() -> Model {
    let gaussian = gaussian_model();
    let rows: Vec<[f64; 2]> = gaussian
        .rows
        .iter()
        .enumerate()
        .map(|(i, r)| [r[0], ((i * 7) % 4) as f64])
        .collect();
    let laws = [
        gaussian.laws[0].clone(),
        Law::Uniform {
            levels: vec![0.0, 1.0, 2.0, 3.0],
        },
    ];
    Model {
        config: gaussian
            .config
            .with_metric(vec![Metric::Euclidean, Metric::Categorical]),
        x: Data::from_rows(&rows).unwrap(),
        rows,
        space: Space::Categorical { weight: 2.0 / 16.0 },
        laws,
        ..gaussian
    }
}

/// The Gaussian model under the Manhattan metric (Minkowski p = 1) on
/// both columns: the calibration rows, the Euclidean coordinate laws,
/// only the assignment of rows to cells changes.
#[cfg(feature = "experimental")]
fn minkowski_model() -> Model {
    let gaussian = gaussian_model();
    Model {
        config: gaussian.config.with_metric(vec![
            Metric::Minkowski { p: 1.0, group: 0 },
            Metric::Minkowski { p: 1.0, group: 0 },
        ]),
        space: Space::Minkowski { p: 1.0 },
        ..gaussian
    }
}

/// The Gaussian model under the cosine distance on both columns: the
/// calibration rows, the Euclidean coordinate laws, only the assignment
/// of rows to cells changes.
#[cfg(feature = "experimental")]
fn cosine_model() -> Model {
    let gaussian = gaussian_model();
    Model {
        config: gaussian.config.with_metric(vec![
            Metric::Cosine { group: 0 },
            Metric::Cosine { group: 0 },
        ]),
        space: Space::Cosine,
        ..gaussian
    }
}

/// The Gaussian model under the Gower distance: the categorical model's
/// rows (a numeric column and four level codes), per-column distances
/// averaged over the active columns.
#[cfg(feature = "experimental")]
fn gower_model() -> Model {
    let categorical = categorical_model();
    Model {
        config: categorical.config.with_metric(vec![
            Metric::Gower {
                kind: GowerKind::Numeric,
                group: 0,
            },
            Metric::Gower {
                kind: GowerKind::Categorical,
                group: 0,
            },
        ]),
        space: Space::Gower,
        ..categorical
    }
}

/// The Gaussian model under the Mahalanobis distance with a fixed
/// correlated precision matrix on both columns: the calibration rows,
/// the Euclidean coordinate laws, only the assignment changes.
#[cfg(feature = "experimental")]
fn mahalanobis_model() -> Model {
    let gaussian = gaussian_model();
    let precision = [2.0, 0.6, 0.6, 1.0];
    Model {
        config: gaussian
            .config
            .with_metric(vec![Metric::Mahalanobis, Metric::Mahalanobis])
            .with_precision(precision.to_vec()),
        space: Space::Mahalanobis { precision },
        ..gaussian
    }
}

/// The Gaussian model under a composite: a Manhattan column of its own
/// group and an Eskin categorical column, on the categorical model's
/// rows and laws.
#[cfg(feature = "experimental")]
fn composite_model() -> Model {
    let categorical = categorical_model();
    Model {
        config: categorical
            .config
            .with_metric(vec![Metric::Manhattan { group: 0 }, Metric::Categorical]),
        space: Space::Composite { weight: 2.0 / 16.0 },
        ..categorical
    }
}

/// The Gaussian model under the weighted inclusion prior, weights
/// (0.75, 0.25): the calibration rows and laws, subsets weighted by the
/// product of member weights, proposals weighted to match.
#[cfg(feature = "experimental")]
fn weighted_model() -> Model {
    let gaussian = gaussian_model();
    let weights = [0.75, 0.25];
    Model {
        config: gaussian.config.with_inclusion(Inclusion::Weighted {
            weights: weights.to_vec(),
        }),
        weights: Some(weights),
        ..gaussian
    }
}

/// The Gaussian model under the DART inclusion prior, a = 0.5, b = 1,
/// rho = 2: the calibration rows and laws, the weights sampled.
#[cfg(feature = "experimental")]
fn dart_model() -> Model {
    let gaussian = gaussian_model();
    Model {
        config: gaussian.config.with_inclusion(Inclusion::Dart {
            a: 0.5,
            b: 1.0,
            rho: Some(2.0),
        }),
        dart: Some((0.5, 1.0, 2.0)),
        quantities: &DART_QUANTITIES,
        n_sbc: 8,
        ..gaussian
    }
}

/// The Gaussian model under the linear cell basis: rows spanning each
/// column's exact scaled range, so the harness's raw coordinates equal
/// the engine's scaled ones, cell slopes N(0, sigma_mu^2).
#[cfg(feature = "experimental")]
fn linear_model() -> Model {
    let gaussian = gaussian_model();
    let n = gaussian.rows.len();
    let rows: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            [
                i as f64 / (n - 1) as f64 - 0.5,
                ((i * 17) % n) as f64 / (n - 1) as f64 - 0.5,
            ]
        })
        .collect();
    Model {
        config: gaussian.config.with_basis(Basis::Linear),
        x: Data::from_rows(&rows).unwrap(),
        rows,
        linear: true,
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
            .with_outcome(thiessen::Outcome::probit())
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
        config: gaussian.config.with_m_var(2),
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
        scale / (self.config.mean_params.k * (self.config.mean_tessellations() as f64).sqrt())
    }

    fn offset(&self) -> f64 {
        self.config.offset().unwrap_or(0.0)
    }

    /// (nu', lambda') of one variance cell.
    fn variance_cell_prior(&self) -> (f64, f64) {
        let root = 1.0 / self.config.variance_tessellations() as f64;
        let (nu, _) = gaussian_prior(&self.config);
        (
            2.0 / (1.0 - (1.0 - 2.0 / nu).powf(root)),
            self.lambda.powf(root),
        )
    }

    /// One tessellation from the prior truncated to full occupancy, by
    /// rejection: structure first, cell values after acceptance (occupancy
    /// does not involve the values).
    fn prior_tessellation(
        &self,
        weights: Option<[f64; 2]>,
        slopes: bool,
        rng: &mut TestRng,
        value: &dyn Fn(&mut TestRng) -> f64,
    ) -> DrawnTessellation {
        let p = self.x.n_cols();
        let theta = self
            .config
            .mean_params
            .structure
            .omega
            .unwrap_or(3.0_f64.min(p as f64))
            / p as f64;
        loop {
            let b = 1 + rng.poisson(self.config.mean_params.lambda_c);
            let mut d = 1;
            for _ in 0..p - 1 {
                if rng.uniform() < theta {
                    d += 1;
                }
            }
            let dims: Vec<usize> = match weights {
                // P(S | d) over two columns: the weight share at d = 1,
                // both columns at d = 2.
                Some(w) if d == 1 => {
                    let target = rng.uniform() * (w[0] + w[1]);
                    vec![usize::from(target >= w[0])]
                }
                Some(_) => vec![0, 1],
                None => {
                    let mut dims: Vec<usize> = (0..p).collect();
                    for i in 0..d {
                        let j = i + (rng.uniform() * (p - i) as f64) as usize;
                        dims.swap(i, j.min(p - 1));
                    }
                    let mut dims: Vec<usize> = dims[..d].to_vec();
                    dims.sort_unstable();
                    dims
                }
            };
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
                let tilts: Vec<f64> = if slopes {
                    let sigma_mu = self.sigma_mu();
                    (0..b * d).map(|_| sigma_mu * rng.normal()).collect()
                } else {
                    Vec::new()
                };
                return (dims, centres, values, tilts);
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
        // theta from the discrete grid, then s ~ Dirichlet(theta / 2):
        // the engine's prior exactly.
        let dart = self.dart.map(|(a, b, rho)| {
            let k = 1000;
            let mut log_prior = Vec::with_capacity(k);
            let mut thetas = Vec::with_capacity(k);
            for i in 1..=k {
                let lambda = i as f64 / (k + 1) as f64;
                thetas.push(lambda * rho / (1.0 - lambda));
                log_prior.push((a - 1.0) * lambda.ln() + (b - 1.0) * (1.0 - lambda).ln());
            }
            let max = log_prior.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
            let w: Vec<f64> = log_prior.iter().map(|&v| (v - max).exp()).collect();
            let total: f64 = w.iter().sum();
            let target = rng.uniform() * total;
            let mut cumulative = 0.0;
            let mut index = k - 1;
            for (i, &v) in w.iter().enumerate() {
                cumulative += v;
                if target < cumulative {
                    index = i;
                    break;
                }
            }
            let theta = thetas[index];
            let g = [
                rng.gamma_any(theta / 2.0).max(f64::MIN_POSITIVE),
                rng.gamma_any(theta / 2.0).max(f64::MIN_POSITIVE),
            ];
            let s = [g[0] / (g[0] + g[1]), g[1] / (g[0] + g[1])];
            (s, theta)
        });
        let weights = dart.map(|(s, _)| s).or(self.weights);
        let tessellations = (0..self.config.mean_tessellations())
            .map(|_| {
                self.prior_tessellation(weights, self.linear, rng, &|rng| sigma_mu * rng.normal())
            })
            .collect();
        let sigma_sq = match self.kind {
            Kind::Gaussian => {
                let chi_sq = -2.0 * (rng.uniform().ln() + rng.uniform().ln() + rng.uniform().ln());
                gaussian_prior(&self.config).0 * self.lambda / chi_sq
            }
            Kind::Probit | Kind::Heteroscedastic => 1.0,
        };
        let variance = match self.kind {
            Kind::Heteroscedastic => {
                let (nu, lambda) = self.variance_cell_prior();
                (0..self.config.variance_tessellations())
                    .map(|_| {
                        self.prior_tessellation(weights, false, rng, &|rng| {
                            0.5 * nu * lambda / rng.gamma(0.5 * nu)
                        })
                    })
                    .collect()
            }
            Kind::Gaussian | Kind::Probit => Vec::new(),
        };
        PriorDraw {
            tessellations,
            sigma_sq,
            variance,
            dart,
        }
    }

    /// The mean ensemble value at training row `i`, tilted by the cell's
    /// slopes under the linear basis.
    fn f_at(&self, draw: &PriorDraw, i: usize) -> f64 {
        let row = &self.rows[i];
        draw.tessellations
            .iter()
            .map(|(dims, centres, mus, tilts)| {
                let k = self.nearest(row, dims, centres);
                let d = dims.len();
                let mut value = mus[k];
                if !tilts.is_empty() {
                    for (j, &dim) in dims.iter().enumerate() {
                        value += tilts[k * d + j] * (row[dim] - centres[k * d + j]);
                    }
                }
                value
            })
            .sum()
    }

    /// The variance of y_i given f at training row `i`.
    fn variance_at(&self, draw: &PriorDraw, i: usize) -> f64 {
        match self.kind {
            Kind::Heteroscedastic => draw
                .variance
                .iter()
                .map(|(dims, centres, values, _)| {
                    values[self.nearest(&self.rows[i], dims, centres)]
                })
                .product(),
            Kind::Gaussian | Kind::Probit => draw.sigma_sq,
        }
    }

    /// Squared distance from `row` to a centre over `dims`: Euclidean, or
    /// the great-circle angle with the centre placed at the row's own
    /// coordinate in an inactive column.
    fn key(&self, row: &[f64; 2], dims: &[usize], centre: &[f64]) -> f64 {
        match self.space {
            Space::Euclidean => {
                return dims
                    .iter()
                    .zip(centre)
                    .map(|(&dim, c)| (row[dim] - c) * (row[dim] - c))
                    .sum();
            }
            Space::Categorical { weight } => {
                return dims
                    .iter()
                    .zip(centre)
                    .map(|(&dim, c)| match dim {
                        0 => (row[0] - c) * (row[0] - c),
                        _ => {
                            if row[1] == *c {
                                0.0
                            } else {
                                weight
                            }
                        }
                    })
                    .sum();
            }
            #[cfg(feature = "experimental")]
            Space::Minkowski { p } => {
                let sum: f64 = dims
                    .iter()
                    .zip(centre)
                    .map(|(&dim, c)| (row[dim] - c).abs().powf(p))
                    .sum();
                return sum.powf(2.0 / p);
            }
            #[cfg(feature = "experimental")]
            Space::Cosine => {
                let (mut dot, mut a, mut b) = (0.0, 0.0, 0.0);
                for (&dim, c) in dims.iter().zip(centre) {
                    dot += row[dim] * c;
                    a += row[dim] * row[dim];
                    b += c * c;
                }
                let d = if a == 0.0 && b == 0.0 {
                    0.0
                } else if a == 0.0 || b == 0.0 {
                    1.0
                } else {
                    (1.0 - dot / (a.sqrt() * b.sqrt())).clamp(0.0, 2.0)
                };
                return d * d;
            }
            #[cfg(feature = "experimental")]
            Space::Gower => {
                let sum: f64 = dims
                    .iter()
                    .zip(centre)
                    .map(|(&dim, c)| match dim {
                        0 => (row[0] - c).abs(),
                        _ => f64::from(row[1] != *c),
                    })
                    .sum();
                let d = sum / dims.len() as f64;
                return d * d;
            }
            #[cfg(feature = "experimental")]
            Space::Mahalanobis { precision } => {
                let mut key = 0.0;
                for (&di, &ci) in dims.iter().zip(centre) {
                    let diff = row[di] - ci;
                    for (&dj, &cj) in dims.iter().zip(centre) {
                        key += diff * precision[di * 2 + dj] * (row[dj] - cj);
                    }
                }
                return key;
            }
            #[cfg(feature = "experimental")]
            Space::Composite { weight } => {
                return dims
                    .iter()
                    .zip(centre)
                    .map(|(&dim, c)| match dim {
                        0 => {
                            let d = (row[0] - c).abs();
                            d * d
                        }
                        _ => {
                            if row[1] == *c {
                                0.0
                            } else {
                                weight
                            }
                        }
                    })
                    .sum();
            }
            Space::Sphere => {}
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
        let cells: usize = draw.tessellations.iter().map(|(_, _, m, _)| m.len()).sum();
        let dims: usize = draw.tessellations.iter().map(|(d, _, _, _)| d.len()).sum();
        let mut out = Vec::with_capacity(self.quantities.len());
        if self.kind == Kind::Gaussian {
            out.push(draw.sigma_sq);
        }
        out.push(cells as f64);
        out.push(dims as f64);
        out.extend(F_ROWS.iter().map(|&r| self.offset() + self.f_at(draw, r)));
        if self.kind == Kind::Heteroscedastic {
            let vcells: usize = draw.variance.iter().map(|(_, _, v, _)| v.len()).sum();
            out.push(vcells as f64);
            out.extend(F_ROWS.iter().map(|&r| self.variance_at(draw, r)));
        }
        if let Some((s, theta)) = draw.dart {
            out.push(s[0]);
            out.push(theta);
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
        #[cfg(feature = "experimental")]
        if self.dart.is_some() {
            let (s, theta) = sampler.inclusion_state().expect("dart state");
            out.push(s[0]);
            out.push(theta);
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

#[test]
fn sbc_small_ranks_are_uniform_categorical() {
    let model = categorical_model();
    let ranks = sbc_ranks(&model, 160, 19, 15, 150, 405);
    assert_uniform(&model, &ranks, 19);
}

#[cfg(feature = "experimental")]
#[test]
fn sbc_small_ranks_are_uniform_minkowski() {
    let model = minkowski_model();
    let ranks = sbc_ranks(&model, 160, 19, 15, 150, 406);
    assert_uniform(&model, &ranks, 19);
}

#[cfg(feature = "experimental")]
#[test]
fn sbc_small_ranks_are_uniform_cosine() {
    let model = cosine_model();
    let ranks = sbc_ranks(&model, 160, 19, 15, 150, 407);
    assert_uniform(&model, &ranks, 19);
}

#[cfg(feature = "experimental")]
#[test]
fn sbc_small_ranks_are_uniform_gower() {
    let model = gower_model();
    let ranks = sbc_ranks(&model, 160, 19, 15, 150, 408);
    assert_uniform(&model, &ranks, 19);
}

#[cfg(feature = "experimental")]
#[test]
fn sbc_small_ranks_are_uniform_mahalanobis() {
    let model = mahalanobis_model();
    let ranks = sbc_ranks(&model, 160, 19, 15, 150, 409);
    assert_uniform(&model, &ranks, 19);
}

#[cfg(feature = "experimental")]
#[test]
fn sbc_small_ranks_are_uniform_composite() {
    let model = composite_model();
    let ranks = sbc_ranks(&model, 160, 19, 15, 150, 410);
    assert_uniform(&model, &ranks, 19);
}

#[cfg(feature = "experimental")]
#[test]
fn sbc_small_ranks_are_uniform_weighted() {
    let model = weighted_model();
    let ranks = sbc_ranks(&model, 160, 19, 15, 150, 411);
    assert_uniform(&model, &ranks, 19);
}

#[cfg(feature = "experimental")]
#[test]
fn sbc_small_ranks_are_uniform_dart() {
    let model = dart_model();
    let ranks = sbc_ranks(&model, 160, 19, 45, 150, 412);
    assert_uniform(&model, &ranks, 19);
}

#[cfg(feature = "experimental")]
#[test]
fn sbc_small_ranks_are_uniform_linear() {
    let model = linear_model();
    let ranks = sbc_ranks(&model, 160, 19, 15, 150, 413);
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

#[test]
#[ignore = "full size, nightly"]
fn sbc_full_ranks_are_uniform_categorical() {
    sbc_full(&categorical_model(), "sbc_ranks_categorical.csv", 405);
}

/// The s chain moves by an independence proposal and needs heavier
/// thinning than the structural quantities, here and in the small run.
#[cfg(feature = "experimental")]
#[test]
#[ignore = "full size, nightly"]
fn sbc_full_ranks_are_uniform_linear() {
    sbc_full(&linear_model(), "sbc_ranks_linear.csv", 413);
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "full size, nightly"]
fn sbc_full_ranks_are_uniform_dart() {
    let model = dart_model();
    let ranks = sbc_ranks(&model, 1000, 99, 60, 300, 412);
    let mut lines = vec!["quantity,rank,max_rank".to_string()];
    for (q, series) in ranks.iter().enumerate() {
        for rank in series {
            lines.push(format!("{},{rank},99", model.quantities[q]));
        }
    }
    write_csv("sbc_ranks_dart.csv", &lines);
    assert_uniform(&model, &ranks, 99);
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

#[test]
fn geweke_small_simulators_agree_categorical() {
    let model = categorical_model();
    let (mc, sc) = geweke_samples(&model, 2000, 800, 45, 200, 911);
    assert_simulators_agree(&model, &mc, &sc);
}

#[cfg(feature = "experimental")]
#[test]
fn geweke_small_simulators_agree_dart() {
    let model = dart_model();
    let (mc, sc) = geweke_samples(&model, 2000, 800, 45, 200, 912);
    assert_simulators_agree(&model, &mc, &sc);
}

#[cfg(feature = "experimental")]
#[test]
fn geweke_small_simulators_agree_linear() {
    let model = linear_model();
    let (mc, sc) = geweke_samples(&model, 2000, 800, 45, 200, 913);
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

#[test]
#[ignore = "full size, nightly"]
fn geweke_full_simulators_agree_categorical() {
    geweke_full(&categorical_model(), "geweke_samples_categorical.csv", 911);
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "full size, nightly"]
fn geweke_full_simulators_agree_dart() {
    geweke_full(&dart_model(), "geweke_samples_dart.csv", 912);
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "full size, nightly"]
fn geweke_full_simulators_agree_linear() {
    geweke_full(&linear_model(), "geweke_samples_linear.csv", 913);
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
