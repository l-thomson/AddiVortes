//! The input-data contract (crate-root documentation, Input data): every
//! rejected input has its error; accepted degenerate inputs have their
//! documented behaviour.

use thiessen::{fit, Config, Data, Error, Warning};

fn config() -> Config {
    Config::new().with_m(3).with_burn_in(5).with_draws(5)
}

fn valid() -> (Data, Vec<f64>) {
    let x = Data::from_rows(&[[1.0, 9.0], [2.0, 7.0], [3.0, 8.0], [4.0, 5.0]]).unwrap();
    (x, vec![0.5, 1.5, 2.5, 3.0])
}

#[test]
fn non_finite_values_are_rejected() {
    let (x, y) = valid();
    let mut bad_y = y.clone();
    bad_y[2] = f64::NAN;
    assert_eq!(
        fit(&config(), &x, &bad_y, 1).unwrap_err(),
        Error::NonFiniteResponse { row: 2 }
    );
    let bad_x =
        Data::from_rows(&[[1.0, 9.0], [2.0, f64::INFINITY], [3.0, 8.0], [4.0, 5.0]]).unwrap();
    assert_eq!(
        fit(&config(), &bad_x, &y, 1).unwrap_err(),
        Error::NonFiniteFeature { row: 1, col: 1 }
    );
}

#[test]
fn shape_mismatches_are_rejected() {
    let (x, _) = valid();
    assert_eq!(
        fit(&config(), &x, &[1.0, 2.0], 1).unwrap_err(),
        Error::RowCountMismatch {
            y_len: 2,
            x_rows: 4
        }
    );
}

#[test]
fn too_few_rows_are_rejected() {
    let x = Data::new(vec![1.0, 2.0], 1, 2).unwrap();
    assert_eq!(
        fit(&config(), &x, &[1.0], 1).unwrap_err(),
        Error::InsufficientObservations {
            found: 1,
            required: 2
        }
    );
}

#[test]
fn zero_columns_are_rejected() {
    let x = Data::new(vec![], 4, 0).unwrap();
    assert_eq!(
        fit(&config(), &x, &[1.0, 2.0, 3.0, 4.0], 1).unwrap_err(),
        Error::NoFeatures
    );
}

#[test]
fn constant_response_and_constant_column_are_rejected() {
    let (x, y) = valid();
    assert_eq!(
        fit(&config(), &x, &[2.0, 2.0, 2.0, 2.0], 1).unwrap_err(),
        Error::DegenerateResponse
    );
    let constant = Data::from_rows(&[[1.0, 6.0], [1.0, 7.0], [1.0, 8.0], [1.0, 5.0]]).unwrap();
    assert_eq!(
        fit(&config(), &constant, &y, 1).unwrap_err(),
        Error::DegenerateFeature { col: 0 }
    );
}

#[test]
fn duplicate_rows_fit() {
    let x = Data::from_rows(&[[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [5.0, 1.0]]).unwrap();
    let model = fit(&config(), &x, &[0.5, 1.5, 2.5, 3.0], 1).unwrap();
    assert_eq!(model.n_draws(), 5);
}

#[test]
fn more_columns_than_rows_fits_with_a_warning() {
    let x = Data::from_rows(&[[1.0, 9.0, 2.0], [2.0, 7.0, 4.0]]).unwrap();
    let model = fit(&config(), &x, &[0.5, 1.5], 1).unwrap();
    assert_eq!(
        model.warnings(),
        &[Warning::MoreFeaturesThanObservations { p: 3, n: 2 }]
    );
}

#[test]
fn predict_checks_columns_and_accepts_an_empty_matrix() {
    let (x, y) = valid();
    let model = fit(&config(), &x, &y, 1).unwrap();
    let narrow = Data::new(vec![1.0, 2.0], 2, 1).unwrap();
    assert_eq!(
        model.predict(&narrow).unwrap_err(),
        Error::FeatureCountMismatch {
            expected: 2,
            found: 1
        }
    );
    let bad = Data::new(vec![f64::NAN, 1.0], 1, 2).unwrap();
    assert!(matches!(
        model.predict(&bad).unwrap_err(),
        Error::NonFiniteFeature { .. }
    ));
    let empty = Data::new(vec![], 0, 2).unwrap();
    assert_eq!(model.predict(&empty).unwrap(), Vec::<f64>::new());
}
