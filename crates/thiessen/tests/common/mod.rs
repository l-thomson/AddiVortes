//! Fixed-seed fixture shared by the determinism and snapshot tests. Every
//! value is exact rational arithmetic, so the data are identical on every
//! target.

use thiessen::{Config, Data};

pub const SEED: u64 = 7;

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
