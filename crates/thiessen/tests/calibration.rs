//! Simulation-based calibration (Talts et al. 2018; Modrák et al. 2025,
//! Bayesian Analysis) and the Geweke (2004) joint-distribution test for
//! the Gaussian model, run under the pinned prior so the prior does not
//! depend on the data.
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
use thiessen::{Config, Data, Sampler};

/// Quantities compared, in column order: sigma^2, total cells and total
/// active dimensions over the ensemble, f at three fixed training rows,
/// and the mean of the generated response. SBC ranks the first six (theta
/// functions); the Geweke test compares all seven.
const QUANTITIES: [&str; 7] = ["sigma_sq", "cells", "dims", "f_a", "f_b", "f_c", "y_mean"];
const F_ROWS: [usize; 3] = [10, 25, 40];

/// Significance 0.01 per test family, Bonferroni-split across the
/// quantities: alpha' = 0.01 / 6 for SBC, 0.01 / 7 for Geweke.
const SBC_CHI2_DF19: f64 = 42.198;
const SBC_CHI2_DF99: f64 = 145.404;
const GEWEKE_CHI2_DF7: f64 = 23.440;
const GEWEKE_CHI2_DF3: f64 = 15.510;
const GEWEKE_ALPHA: f64 = 0.01 / 7.0;

struct Model {
    config: Config,
    lambda: f64,
    x: Data,
    rows: Vec<[f64; 2]>,
}

/// One prior draw of the ensemble, engine-free: the test's own sampler.
struct PriorDraw {
    /// Per tessellation: active columns, row-major centres, cell means.
    tessellations: Vec<(Vec<usize>, Vec<f64>, Vec<f64>)>,
    sigma_sq: f64,
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
    Model {
        config,
        lambda: 0.04,
        x,
        rows,
    }
}

impl Model {
    fn sigma_mu(&self) -> f64 {
        0.5 / (self.config.k * (self.config.m as f64).sqrt())
    }

    /// One tessellation from the prior truncated to full occupancy, by
    /// rejection: structure first, means after acceptance (occupancy does
    /// not involve the means).
    fn prior_tessellation(&self, rng: &mut TestRng) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
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
            let centres: Vec<f64> = (0..b * d)
                .map(|_| self.config.sigma_c * rng.normal())
                .collect();
            let mut occupied = vec![false; b];
            for row in &self.rows {
                occupied[nearest(row, &dims, &centres, d)] = true;
            }
            if occupied.iter().all(|&o| o) {
                let mus: Vec<f64> = (0..b).map(|_| self.sigma_mu() * rng.normal()).collect();
                return (dims, centres, mus);
            }
        }
    }

    /// theta ~ prior: m truncated tessellations and sigma^2 from
    /// nu lambda / chi^2_nu (nu = 6: the chi-squared is a sum of three
    /// exponentials).
    fn prior_draw(&self, rng: &mut TestRng) -> PriorDraw {
        let tessellations = (0..self.config.m)
            .map(|_| self.prior_tessellation(rng))
            .collect();
        let chi_sq = -2.0 * (rng.uniform().ln() + rng.uniform().ln() + rng.uniform().ln());
        PriorDraw {
            tessellations,
            sigma_sq: self.config.nu * self.lambda / chi_sq,
        }
    }

    /// The ensemble value at training row `i`.
    fn f_at(&self, draw: &PriorDraw, i: usize) -> f64 {
        draw.tessellations
            .iter()
            .map(|(dims, centres, mus)| mus[nearest(&self.rows[i], dims, centres, dims.len())])
            .sum()
    }

    /// y | theta.
    fn generate_y(&self, draw: &PriorDraw, rng: &mut TestRng) -> Vec<f64> {
        let sigma = draw.sigma_sq.sqrt();
        (0..self.x.n_rows())
            .map(|i| self.f_at(draw, i) + sigma * rng.normal())
            .collect()
    }

    /// The seven quantities of a marginal-conditional draw.
    fn mc_quantities(&self, draw: &PriorDraw, y: &[f64]) -> [f64; 7] {
        let cells: usize = draw.tessellations.iter().map(|(_, _, m)| m.len()).sum();
        let dims: usize = draw.tessellations.iter().map(|(d, _, _)| d.len()).sum();
        [
            draw.sigma_sq,
            cells as f64,
            dims as f64,
            self.f_at(draw, F_ROWS[0]),
            self.f_at(draw, F_ROWS[1]),
            self.f_at(draw, F_ROWS[2]),
            y.iter().sum::<f64>() / y.len() as f64,
        ]
    }
}

/// Nearest centre of `centres` (row-major, `d` coordinates each) to `row`
/// over the columns `dims`; ties to the lowest index, matching the engine.
fn nearest(row: &[f64], dims: &[usize], centres: &[f64], d: usize) -> usize {
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

/// The seven quantities of the current sampler state and the response it
/// conditioned on.
fn sampler_quantities(sampler: &Sampler, y: &[f64]) -> [f64; 7] {
    let cells: usize = sampler.tessellations().iter().map(|t| t.n_cells()).sum();
    let dims: usize = sampler.tessellations().iter().map(|t| t.n_dims()).sum();
    let fit = sampler.fitted_values();
    [
        sampler.sigma_sq(),
        cells as f64,
        dims as f64,
        fit[F_ROWS[0]],
        fit[F_ROWS[1]],
        fit[F_ROWS[2]],
        y.iter().sum::<f64>() / y.len() as f64,
    ]
}

// ---------------------------------------------------------------------------
// Simulation-based calibration
// ---------------------------------------------------------------------------

/// Ranks of the first six quantities over `sims` simulations: theta from
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
    let mut ranks: Vec<Vec<usize>> = (0..6).map(|_| Vec::with_capacity(sims)).collect();
    for sim in 0..sims {
        let mut rng = TestRng(seed ^ (sim as u64).wrapping_mul(0x9E37_79B9));
        let draw = model.prior_draw(&mut rng);
        let y = model.generate_y(&draw, &mut rng);
        let truth = model.mc_quantities(&draw, &y);
        let mut sampler =
            Sampler::pinned_prior(&model.config, &model.x, &y, model.lambda, seed + sim as u64)
                .unwrap();
        for _ in 0..burn_in {
            sampler.step();
        }
        let mut draws: Vec<Vec<f64>> = (0..6).map(|_| Vec::with_capacity(kept)).collect();
        for _ in 0..kept {
            for _ in 0..thinning {
                sampler.step();
            }
            let state = sampler_quantities(&sampler, &y);
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

fn assert_uniform(ranks: &[Vec<usize>], kept: usize, critical: f64) {
    for (q, series) in ranks.iter().enumerate() {
        let statistic = rank_uniformity(series, kept);
        assert!(
            statistic < critical,
            "{}: chi-squared {statistic} over {} bins, critical {critical}",
            QUANTITIES[q],
            kept + 1
        );
    }
}

#[test]
fn sbc_small_ranks_are_uniform() {
    let ranks = sbc_ranks(&gaussian_model(), 160, 19, 15, 150, 401);
    assert_uniform(&ranks, 19, SBC_CHI2_DF19);
}

#[test]
#[ignore = "full size, nightly"]
fn sbc_full_ranks_are_uniform() {
    let ranks = sbc_ranks(&gaussian_model(), 1000, 99, 20, 300, 401);
    // Files first, so a failed gate still leaves the R evaluation its
    // input.
    let mut lines = vec!["quantity,rank,max_rank".to_string()];
    for (q, series) in ranks.iter().enumerate() {
        for rank in series {
            lines.push(format!("{},{rank},99", QUANTITIES[q]));
        }
    }
    write_csv("sbc_ranks.csv", &lines);
    assert_uniform(&ranks, 99, SBC_CHI2_DF99);
}

// ---------------------------------------------------------------------------
// Geweke joint-distribution test
// ---------------------------------------------------------------------------

/// Marginal-conditional against successive-conditional samples per
/// quantity (Geweke 2004). The successive-conditional chain alternates
/// y | theta (through `set_response`) with theta | y (one sweep of the
/// kernel under test), discards `discard` transitions and keeps every
/// `thin`-th of the next `n_sc * thin`.
fn geweke_samples(
    model: &Model,
    n_mc: usize,
    n_sc: usize,
    thin: usize,
    discard: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut rng = TestRng(seed);
    let mut mc: Vec<Vec<f64>> = (0..7).map(|_| Vec::with_capacity(n_mc)).collect();
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
    let mut sc: Vec<Vec<f64>> = (0..7).map(|_| Vec::with_capacity(n_sc)).collect();
    let transition = |sampler: &mut Sampler, y: &mut Vec<f64>, rng: &mut TestRng| {
        let fit = sampler.fitted_values();
        let sigma = sampler.sigma_sq().sqrt();
        for (slot, f) in y.iter_mut().zip(&fit) {
            *slot = f + sigma * rng.normal();
        }
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
        let state = sampler_quantities(&sampler, &y);
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

fn assert_simulators_agree(mc: &[Vec<f64>], sc: &[Vec<f64>]) {
    for (q, name) in QUANTITIES.iter().enumerate() {
        match *name {
            // Total cells over m = 3 tessellations, bins <=5 to >=12;
            // total dimensions, bins 3 to 6.
            "cells" => {
                let statistic = chi_squared_binned(&mc[q], &sc[q], 5, 12);
                assert!(
                    statistic < GEWEKE_CHI2_DF7,
                    "cells: chi-squared {statistic}, critical {GEWEKE_CHI2_DF7}"
                );
            }
            "dims" => {
                let statistic = chi_squared_binned(&mc[q], &sc[q], 3, 6);
                assert!(
                    statistic < GEWEKE_CHI2_DF3,
                    "dims: chi-squared {statistic}, critical {GEWEKE_CHI2_DF3}"
                );
            }
            _ => {
                let d = ks_statistic(&mc[q], &sc[q]);
                let c = (-(GEWEKE_ALPHA / 2.0).ln() / 2.0).sqrt();
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
    let (mc, sc) = geweke_samples(&gaussian_model(), 2000, 800, 15, 200, 907);
    assert_simulators_agree(&mc, &sc);
}

#[test]
#[ignore = "full size, nightly"]
fn geweke_full_simulators_agree() {
    let (mc, sc) = geweke_samples(&gaussian_model(), 20_000, 5000, 20, 500, 907);
    // Files first, so a failed gate still leaves the R evaluation its
    // input.
    let mut lines = vec!["quantity,simulator,value".to_string()];
    for (q, name) in QUANTITIES.iter().enumerate() {
        for v in &mc[q] {
            lines.push(format!("{name},mc,{v}"));
        }
        for v in &sc[q] {
            lines.push(format!("{name},sc,{v}"));
        }
    }
    write_csv("geweke_samples.csv", &lines);
    assert_simulators_agree(&mc, &sc);
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
