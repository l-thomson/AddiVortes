//! Test support shared across the integration tests: the fixed-seed
//! fixture of the determinism and snapshot tests, and a self-contained
//! generator for reference simulators, which must share nothing with the
//! crate's RNG.

use thiessen::{Config, Data};

#[allow(dead_code)]
pub const SEED: u64 = 7;

/// splitmix64 (Steele, Lea and Flood 2014) with Box-Muller normals and
/// inversion sampling for the counts.
pub struct TestRng(pub u64);

#[allow(dead_code)]
impl TestRng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }

    pub fn normal(&mut self) -> f64 {
        let (u1, u2) = (1.0 - self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn poisson(&mut self, lambda: f64) -> usize {
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

#[allow(dead_code)]
pub fn fixture() -> (Config, Data, Vec<f64>) {
    let n = 48;
    let rows: Vec<[f64; 2]> = (0..n)
        .map(|i| [i as f64 / (n - 1) as f64, ((i * 37) % n) as f64 / n as f64])
        .collect();
    let y: Vec<f64> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let f = 3.0 * (r[0] - 0.4) * (r[0] - 0.4) + 0.5 * r[1];
            f + 0.3 * (((i * 29) % 17) as f64 / 16.0 - 0.5)
        })
        .collect();
    let x = Data::from_rows(&rows).unwrap();
    let config = Config::new().with_m(15).with_burn_in(50).with_draws(60);
    (config, x, y)
}
