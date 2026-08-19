//! Fits the Gaussian model end to end on synthetic data and prints
//! posterior summaries.

use thiessen::{fit, Config, Data};

fn main() -> thiessen::Result<()> {
    let n = 120;
    let mut state = 1u64;
    let mut noise = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let rows: Vec<[f64; 2]> = (0..n)
        .map(|i| [i as f64 / (n - 1) as f64, ((i * 53) % n) as f64 / n as f64])
        .collect();
    let y: Vec<f64> = rows
        .iter()
        .map(|r| (4.0 * r[0]).sin() + r[1] * r[1] + 0.2 * noise())
        .collect();
    let x = Data::from_rows(&rows)?;

    let config = Config::new().with_m(50).with_burn_in(100).with_draws(200);
    let model = fit(&config, &x, &y, 1)?;

    let sigma = model.sigma();
    println!("kept draws: {}", model.n_draws());
    println!("in-sample RMSE: {:.3}", model.in_sample_rmse());
    println!(
        "posterior mean sigma: {:.3}",
        sigma.iter().sum::<f64>() / sigma.len() as f64
    );
    let interval = model.credible_interval(&x, 0.95)?;
    println!(
        "95% credible interval for f at the first row: [{:.3}, {:.3}]",
        interval[0].lower, interval[0].upper
    );
    Ok(())
}
