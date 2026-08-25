//! The benchmark registry: one workload per shipped model, at a fixed
//! size, with the deterministic generators behind it. A model is
//! benchmarked because it appears in [`CASES`]; nothing else has to be
//! edited when one is added.
//!
//! The generators share nothing with the crate's RNG, so a change to the
//! sampler's RNG cannot silently change the data a benchmark runs on.
//!
//! Sizes are fixed here and nowhere else. The mean ensemble, the priors
//! and the geometry are the shipped defaults, so the benchmarks track the
//! configuration a user gets rather than one tuned to be fast.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use thiessen::{Config, Data, Fitted, Metric, Outcome, Sampler};

/// Rows in every registry workload.
pub const N: usize = 200;

/// Covariate columns in every registry workload.
pub const P: usize = 10;

/// Burn-in sweeps for the fitted models the predict benchmarks use. Short:
/// the predict cost is a function of the kept draws and the ensemble size,
/// not of how well the chain has converged.
pub const BURN_IN: usize = 20;

/// Kept draws for the fitted models the predict benchmarks use.
pub const DRAWS: usize = 50;

/// Seed for every registry workload.
pub const SEED: u64 = 20_260_824;

/// Column counts of the scaling benchmark. A tessellation reads a handful
/// of columns, so its cost is near-flat in p; a fit that grows in p is a
/// defect in the distance path rather than a property of the method.
pub const SCALING_P: [usize; 3] = [5, 10, 40];

/// splitmix64 (Steele, Lea and Flood 2014), with Box-Muller normals.
pub struct Rng(u64);

impl Rng {
    /// A generator keyed by `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// U(0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }

    /// N(0, 1).
    pub fn normal(&mut self) -> f64 {
        let (u1, u2) = (1.0 - self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Friedman (1991) benchmark #1 at `n` rows and `p` columns: the design,
/// the noiseless mean and the standard normal noise, kept apart so a model
/// can rescale the noise or threshold the mean.
///
/// # Panics
///
/// If `p` is below 5, the number of covariates the function reads.
pub fn friedman(n: usize, p: usize, seed: u64) -> (Data, Vec<f64>, Vec<f64>) {
    assert!(p >= 5, "Friedman #1 reads five covariates");
    let mut rng = Rng::new(seed);
    let values: Vec<f64> = (0..n * p).map(|_| rng.uniform()).collect();
    let x = Data::new(values, n, p).expect("registry design is valid");
    let mean: Vec<f64> = (0..n)
        .map(|i| {
            let r = x.row(i);
            10.0 * (std::f64::consts::PI * r[0] * r[1]).sin()
                + 20.0 * (r[2] - 0.5).powi(2)
                + 10.0 * r[3]
                + 5.0 * r[4]
        })
        .collect();
    let noise: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    (x, mean, noise)
}

/// The response a model is fitted to. The variants are the crate's three
/// fit entry points, so a registry case names its data shape and the
/// dispatch below stays exhaustive.
pub enum Response {
    /// A plain numeric response: [`thiessen::fit`].
    Numeric(Vec<f64>),
    /// Event or censoring times with an event indicator:
    /// [`thiessen::fit_aft`].
    #[cfg(feature = "experimental")]
    Survival {
        /// One positive time per row.
        times: Vec<f64>,
        /// True for an event, false for right-censoring.
        events: Vec<bool>,
    },
    /// One pair of bounds per row: [`thiessen::fit_interval_censored`].
    #[cfg(feature = "experimental")]
    Bounds {
        /// Lower bounds, one per row.
        lower: Vec<f64>,
        /// Upper bounds, one per row.
        upper: Vec<f64>,
    },
}

/// A configuration, a design and a response: everything a benchmark needs
/// to start a sampler or produce a fitted model.
pub struct Workload {
    /// The configuration under benchmark.
    pub config: Config,
    /// The design.
    pub x: Data,
    /// The response, in the shape the model's fit entry point takes.
    pub response: Response,
}

impl Workload {
    /// A sampler on this workload, before any sweep.
    ///
    /// # Panics
    ///
    /// If the workload fails validation, which is a defect in the registry.
    pub fn sampler(&self, seed: u64) -> Sampler {
        let built = match &self.response {
            Response::Numeric(y) => Sampler::new(&self.config, &self.x, y, seed),
            #[cfg(feature = "experimental")]
            Response::Survival { times, events } => {
                Sampler::aft(&self.config, &self.x, times, events, seed)
            }
            #[cfg(feature = "experimental")]
            Response::Bounds { lower, upper } => {
                Sampler::interval_censored(&self.config, &self.x, lower, upper, seed)
            }
        };
        built.expect("registry workload is valid")
    }

    /// The numeric response, for the benchmarks that call
    /// [`thiessen::fit`] directly.
    ///
    /// # Panics
    ///
    /// If the workload carries a censored response.
    pub fn numeric(&self) -> &[f64] {
        match &self.response {
            Response::Numeric(y) => y,
            #[cfg(feature = "experimental")]
            _ => panic!("workload has no numeric response"),
        }
    }

    /// This workload fitted to the schedule its configuration carries.
    ///
    /// # Panics
    ///
    /// If the workload fails validation, which is a defect in the registry.
    pub fn fit(&self, seed: u64) -> Fitted {
        let fitted = match &self.response {
            Response::Numeric(y) => thiessen::fit(&self.config, &self.x, y, seed),
            #[cfg(feature = "experimental")]
            Response::Survival { times, events } => {
                thiessen::fit_aft(&self.config, &self.x, times, events, seed)
            }
            #[cfg(feature = "experimental")]
            Response::Bounds { lower, upper } => {
                thiessen::fit_interval_censored(&self.config, &self.x, lower, upper, seed)
            }
        };
        fitted.expect("registry workload is valid")
    }
}

/// A registry entry: the model's name, as [`Fitted::model_name`] reports
/// it, and the workload builder behind it.
pub struct Case {
    /// The model name; the benchmark identifier is built from it.
    pub name: &'static str,
    /// Builds the workload at `n` rows and `p` columns.
    pub build: fn(usize, usize) -> Workload,
}

/// Every shipped model, stable first. Adding a model here is what puts it
/// under benchmark.
pub const CASES: &[Case] = &[
    Case {
        name: "gaussian",
        build: gaussian,
    },
    Case {
        name: "probit",
        build: probit,
    },
    Case {
        name: "heteroscedastic",
        build: heteroscedastic,
    },
    #[cfg(feature = "experimental")]
    Case {
        name: "tobit",
        build: tobit,
    },
    #[cfg(feature = "experimental")]
    Case {
        name: "ordinal",
        build: ordinal,
    },
    #[cfg(feature = "experimental")]
    Case {
        name: "aft",
        build: aft,
    },
    #[cfg(feature = "experimental")]
    Case {
        name: "interval_censored",
        build: interval_censored,
    },
    #[cfg(feature = "experimental")]
    Case {
        name: "student_t",
        build: student_t,
    },
    #[cfg(feature = "experimental")]
    Case {
        name: "laplace",
        build: laplace,
    },
];

/// The schedule the predict benchmarks fit under; the sweep benchmarks
/// ignore it and step the sampler directly.
fn schedule(config: Config) -> Config {
    config.with_burn_in(BURN_IN).with_draws(DRAWS)
}

fn gaussian(n: usize, p: usize) -> Workload {
    let (x, mean, noise) = friedman(n, p, SEED);
    let y = mean.iter().zip(&noise).map(|(m, e)| m + e).collect();
    Workload {
        config: schedule(Config::new()),
        x,
        response: Response::Numeric(y),
    }
}

fn probit(n: usize, p: usize) -> Workload {
    let (x, mean, _) = friedman(n, p, SEED);
    let cut = quantile(&mean, 0.5);
    let y = mean.iter().map(|&m| f64::from(m >= cut)).collect();
    Workload {
        config: schedule(Config::new().with_outcome(Outcome::probit())),
        x,
        response: Response::Numeric(y),
    }
}

fn heteroscedastic(n: usize, p: usize) -> Workload {
    let (x, mean, noise) = friedman(n, p, SEED);
    let y = (0..n)
        .map(|i| mean[i] + noise[i] * (0.2 + 2.0 * x.row(i)[0]))
        .collect();
    Workload {
        // Twenty variance tessellations against the mean ensemble's
        // default: the ratio the H-AddiVortes examples use.
        config: schedule(Config::new().with_m_var(20)),
        x,
        response: Response::Numeric(y),
    }
}

#[cfg(feature = "experimental")]
fn student_t(n: usize, p: usize) -> Workload {
    let (x, mean, noise) = friedman(n, p, SEED);
    // Every twentieth row carries a fivefold error, the outliers the
    // weights are there to discount.
    let y = mean
        .iter()
        .zip(&noise)
        .enumerate()
        .map(|(i, (m, e))| if i % 20 == 0 { m + 5.0 * e } else { m + e })
        .collect();
    Workload {
        config: schedule(Config::new().with_outcome(Outcome::student_t(4.0))),
        x,
        response: Response::Numeric(y),
    }
}

#[cfg(feature = "experimental")]
fn laplace(n: usize, p: usize) -> Workload {
    let (x, mean, noise) = friedman(n, p, SEED);
    // The Student-t workload's outliers, under the Laplace model.
    let y = mean
        .iter()
        .zip(&noise)
        .enumerate()
        .map(|(i, (m, e))| if i % 20 == 0 { m + 5.0 * e } else { m + e })
        .collect();
    Workload {
        config: schedule(Config::new().with_outcome(Outcome::laplace())),
        x,
        response: Response::Numeric(y),
    }
}

#[cfg(feature = "experimental")]
fn tobit(n: usize, p: usize) -> Workload {
    let (x, mean, noise) = friedman(n, p, SEED);
    let latent: Vec<f64> = mean.iter().zip(&noise).map(|(m, e)| m + e).collect();
    // A quarter of the rows at the limit: enough augmentation to dominate
    // nothing and enough to be exercised.
    let lower = quantile(&latent, 0.25);
    let y = latent.iter().map(|&v| v.max(lower)).collect();
    Workload {
        config: schedule(Config::new().with_outcome(Outcome::tobit(Some(lower), None))),
        x,
        response: Response::Numeric(y),
    }
}

#[cfg(feature = "experimental")]
fn ordinal(n: usize, p: usize) -> Workload {
    let (x, mean, noise) = friedman(n, p, SEED);
    let latent: Vec<f64> = mean.iter().zip(&noise).map(|(m, e)| m + e).collect();
    let cuts = [quantile(&latent, 1.0 / 3.0), quantile(&latent, 2.0 / 3.0)];
    let y = latent
        .iter()
        .map(|&v| cuts.iter().filter(|&&c| v >= c).count() as f64)
        .collect();
    Workload {
        config: schedule(Config::new().with_outcome(Outcome::ordinal(cuts.len() + 1))),
        x,
        response: Response::Numeric(y),
    }
}

#[cfg(feature = "experimental")]
fn aft(n: usize, p: usize) -> Workload {
    let (x, mean, noise) = friedman(n, p, SEED);
    // The mean is on the log scale, scaled down so the times stay in a
    // range a double represents without loss.
    let times: Vec<f64> = (0..n)
        .map(|i| (mean[i] / 10.0 + noise[i] / 5.0).exp())
        .collect();
    // Every third row right-censored, which is the censoring share the
    // `abart` comparison in the variant tests uses.
    let events = (0..n).map(|i| i % 3 != 0).collect();
    Workload {
        config: schedule(Config::new().with_outcome(Outcome::aft())),
        x,
        response: Response::Survival { times, events },
    }
}

#[cfg(feature = "experimental")]
fn interval_censored(n: usize, p: usize) -> Workload {
    let (x, mean, noise) = friedman(n, p, SEED);
    let latent: Vec<f64> = mean.iter().zip(&noise).map(|(m, e)| m + e).collect();
    // Alternating exact values and unit-width brackets, so both branches
    // of the augmentation are on the path.
    let lower = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                latent[i]
            } else {
                latent[i] - 0.5
            }
        })
        .collect();
    let upper = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                latent[i]
            } else {
                latent[i] + 0.5
            }
        })
        .collect();
    Workload {
        config: schedule(Config::new().with_outcome(Outcome::interval_censored())),
        x,
        response: Response::Bounds { lower, upper },
    }
}

/// The scaling workload: the Gaussian model at `p` columns, used to check
/// that the fit stays near-flat in p.
pub fn scaling(p: usize) -> Workload {
    gaussian(N, p)
}

/// The spherical workload: the Gaussian model over one sphere, so the
/// great-circle path is measured beside the Euclidean one.
pub fn spherical(n: usize) -> Workload {
    use std::f64::consts::PI;
    let mut rng = Rng::new(SEED);
    let rows: Vec<[f64; 2]> = (0..n)
        .map(|_| [(rng.uniform() - 0.5) * PI, (rng.uniform() - 0.5) * 2.0 * PI])
        .collect();
    let y = rows
        .iter()
        .map(|r| 5.0 * r[0].cos() * r[1].cos() + 3.0 * r[0].sin() + rng.normal() / 2.0)
        .collect();
    let x = Data::from_rows(&rows).expect("registry design is valid");
    Workload {
        config: schedule(Config::new().with_metric(vec![
            Metric::Spherical { sphere: 0 },
            Metric::Spherical { sphere: 0 },
        ])),
        x,
        response: Response::Numeric(y),
    }
}

/// Sizes of the binding-overhead cases. Three, because a copy of the
/// design shows as a slope and not as an offset: one baseline, one that
/// grows in rows, one that grows in columns.
pub const BINDING_SIZES: &[(usize, usize)] = &[(200, 10), (2000, 10), (200, 40)];

/// Rows in the predict matrix of the binding-overhead cases, large enough
/// that marshalling the return value is visible beside the traversal.
pub const PREDICT_ROWS: usize = 5000;

/// Sweeps in the per-call binding case: run behind one crossing and behind
/// one crossing each, the difference being the boundary cost.
pub const BINDING_SWEEPS: usize = 200;

/// The suite cell sizes, small first. Two sizes so a cost that grows in n
/// or p is separable from one that does not.
pub const CELL_SIZES: &[(usize, usize)] = &[(200, 10), (1000, 20)];

/// Held-out rows per cell, for the accuracy metrics of the scorecard.
pub const HOLDOUT: usize = 100;

/// Held-out rows whose posterior f(x) draws are declared quantities of the
/// scorecard. Kept small: each is a separate series through the diagnostic
/// estimator.
pub const DECLARED_ROWS: usize = 5;

/// Burn-in sweeps in a suite cell.
pub const CELL_BURN_IN: usize = 500;

/// Kept draws in a suite cell. Longer than the shipped default of 1000,
/// which leaves rank-normalised R-hat around 1.035 on held-out f(x) for
/// the Gaussian model at these sizes: efficiency measured on a chain that
/// has not converged describes nothing, so the suite pays for convergence
/// and reports the cost as part of the measurement.
pub const CELL_DRAWS: usize = 4000;

/// One cell of the benchmark suite: a model, a size and a seed.
pub struct Cell {
    /// The registry model name.
    pub model: &'static str,
    /// Training rows.
    pub n: usize,
    /// Covariate columns.
    pub p: usize,
}

impl Cell {
    /// The cell's identifier, as the scorecard and the output files use it.
    pub fn id(&self) -> String {
        format!("{}-n{}-p{}", self.model, self.n, self.p)
    }
}

/// Every cell of the suite: each registry model at each declared size.
pub fn cells() -> Vec<Cell> {
    CASES
        .iter()
        .flat_map(|case| {
            CELL_SIZES.iter().map(move |&(n, p)| Cell {
                model: case.name,
                n,
                p,
            })
        })
        .collect()
}

/// A cell's training workload with the held-out rows kept back.
pub struct Split {
    /// The training workload, on the suite's schedule.
    pub train: Workload,
    /// The held-out design.
    pub test_x: Data,
    /// The held-out response, absent where the model's response is
    /// censored and there is no value to score a prediction against.
    pub test_y: Option<Vec<f64>>,
}

/// Build `cell`: the registry workload at `cell.n + HOLDOUT` rows, split
/// into the training set and the held-out rows, on the suite's schedule.
///
/// # Panics
///
/// If the registry carries no model of that name.
pub fn build_cell(cell: &Cell) -> Split {
    let case = CASES
        .iter()
        .find(|c| c.name == cell.model)
        .unwrap_or_else(|| panic!("no registry case named {}", cell.model));
    let full = (case.build)(cell.n + HOLDOUT, cell.p);
    let train_rows = cell.n;
    let head = |x: &Data| rows_of(x, 0, train_rows);
    let tail = |x: &Data| rows_of(x, train_rows, x.n_rows());
    let (test_x, train_x) = (tail(&full.x), head(&full.x));
    let (response, test_y) = match full.response {
        Response::Numeric(y) => (
            Response::Numeric(y[..train_rows].to_vec()),
            Some(y[train_rows..].to_vec()),
        ),
        #[cfg(feature = "experimental")]
        Response::Survival { times, events } => (
            Response::Survival {
                times: times[..train_rows].to_vec(),
                events: events[..train_rows].to_vec(),
            },
            None,
        ),
        #[cfg(feature = "experimental")]
        Response::Bounds { lower, upper } => (
            Response::Bounds {
                lower: lower[..train_rows].to_vec(),
                upper: upper[..train_rows].to_vec(),
            },
            None,
        ),
    };
    Split {
        train: Workload {
            config: full
                .config
                .with_burn_in(CELL_BURN_IN)
                .with_draws(CELL_DRAWS),
            x: train_x,
            response,
        },
        test_x,
        test_y,
    }
}

/// Rows `start..end` of `x` as a design of their own.
fn rows_of(x: &Data, start: usize, end: usize) -> Data {
    let p = x.n_cols();
    let mut values = Vec::with_capacity((end - start) * p);
    for i in start..end {
        values.extend_from_slice(x.row(i));
    }
    Data::new(values, end - start, p).expect("a row subset of a valid design is valid")
}

/// The type 7 quantile of `values`.
fn quantile(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let h = (sorted.len() as f64 - 1.0) * p;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}
