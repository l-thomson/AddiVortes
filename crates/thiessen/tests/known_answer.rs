//! Known-answer tests through the public surface: the pinned prior, and
//! prior recovery against independent rejection sampling.

use thiessen::{fit, Config, Data, Sampler};

/// splitmix64 with Box-Muller, self-contained so the reference sampler
/// shares nothing with the crate's RNG.
struct TestRng(u64);

impl TestRng {
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

#[test]
fn pinned_prior_is_in_force() {
    let n = 20;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 - 0.5).collect();
    let y: Vec<f64> = (0..n).map(|i| ((i * 3) % 7) as f64 / 14.0 - 0.25).collect();
    let x = Data::new(xs, n, 1).unwrap();
    let config = Config::new().with_m(4);

    let pinned = Sampler::pinned_prior(&config, &x, &y, 0.03, 5).unwrap();
    assert_eq!(pinned.lambda(), 0.03);
    assert_eq!(pinned.scaler().y_min(), -0.5);
    assert_eq!(pinned.scaler().y_max(), 0.5);
    assert_eq!(pinned.scaler().x_min(), &[-0.5]);
    assert_eq!(pinned.scaler().x_max(), &[0.5]);

    let calibrated = Sampler::new(&config, &x, &y, 5).unwrap();
    assert_ne!(calibrated.lambda(), 0.03);

    assert!(Sampler::pinned_prior(&config, &x, &y, 0.0, 5).is_err());
    assert!(Sampler::pinned_prior(&config, &x, &y, f64::NAN, 5).is_err());
}

/// The prior-only chain and an independent rejection sampler target the
/// same distribution: the structural prior truncated to tessellations
/// whose cells all hold a training row.
#[test]
fn prior_only_draws_match_independent_rejection_sampling() {
    let n = 60;
    let rows: Vec<[f64; 2]> = (0..n)
        .map(|i| [i as f64 / (n - 1) as f64, ((i * 23) % n) as f64 / n as f64])
        .collect();
    let y: Vec<f64> = (0..n).map(|i| ((i * 7) % n) as f64 / n as f64).collect();
    let x = Data::from_rows(&rows).unwrap();

    let (lambda_c, omega, sigma_c) = (2.0, 0.8, 0.8);
    let config = Config::new()
        .with_prior_only(true)
        .with_m(1)
        .with_omega(omega)
        .with_lambda_c(lambda_c)
        .with_sigma_c(sigma_c)
        .with_burn_in(500)
        .with_draws(2000)
        .with_thinning(20);
    let fitted = fit(&config, &x, &y, 31).unwrap();
    let chain: Vec<(usize, usize)> = fitted
        .posterior()
        .tessellations()
        .iter()
        .map(|draw| (draw[0].n_cells(), draw[0].n_dims()))
        .collect();

    // The scaled rows the occupancy rule sees, per column
    // (v - min) / (max - min) - 0.5.
    let scaled: Vec<[f64; 2]> = {
        let (mut min, mut max) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for r in &rows {
            for c in 0..2 {
                min[c] = min[c].min(r[c]);
                max[c] = max[c].max(r[c]);
            }
        }
        rows.iter()
            .map(|r| [0, 1].map(|c| (r[c] - min[c]) / (max[c] - min[c]) - 0.5))
            .collect()
    };

    let mut rng = TestRng(97);
    let theta = omega / 2.0;
    let mut reference = Vec::with_capacity(4000);
    while reference.len() < 4000 {
        let b = 1 + rng.poisson(lambda_c);
        let d = if rng.uniform() < theta { 2 } else { 1 };
        let dims: Vec<usize> = if d == 2 {
            vec![0, 1]
        } else if rng.uniform() < 0.5 {
            vec![0]
        } else {
            vec![1]
        };
        let centres: Vec<f64> = (0..b * d).map(|_| sigma_c * rng.normal()).collect();
        let mut occupied = vec![false; b];
        for row in &scaled {
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
            reference.push((b, d));
        }
    }

    // Joint bins over (min(b, 6), d), 12 cells; two-sample chi-squared
    // statistic against the 0.999 quantile of chi^2_11, 31.264.
    let bin = |(b, d): &(usize, usize)| -> usize { (b.min(&6) - 1) * 2 + (d - 1) };
    let mut counts = [[0.0_f64; 12]; 2];
    for value in &chain {
        counts[0][bin(value)] += 1.0;
    }
    for value in &reference {
        counts[1][bin(value)] += 1.0;
    }
    let (n1, n2) = (chain.len() as f64, reference.len() as f64);
    let (r1, r2) = ((n2 / n1).sqrt(), (n1 / n2).sqrt());
    let mut statistic = 0.0;
    for (i, (&c1, &c2)) in counts[0].iter().zip(&counts[1]).enumerate() {
        assert!(c1 + c2 > 0.0, "empty bin {i}");
        let diff = r1 * c1 - r2 * c2;
        statistic += diff * diff / (c1 + c2);
    }
    assert!(statistic < 31.264, "chi-squared {statistic}");
}
