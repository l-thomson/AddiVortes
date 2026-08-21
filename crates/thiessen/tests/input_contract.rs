//! The input-data contract (crate-root documentation, Input data): every
//! rejected input has its error; accepted degenerate inputs have their
//! documented behaviour.

use thiessen::{fit, Config, Data, Error, Metric, Warning};

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
fn a_single_feature_fits() {
    let n = 8;
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = xs.iter().map(|v| v * v).collect();
    let x = Data::new(xs, n, 1).unwrap();
    let model = fit(&config(), &x, &y, 1).unwrap();
    assert_eq!(model.predict(&x).unwrap().len(), n);
}

#[test]
fn zero_residual_response_fits() {
    // y exactly linear in x: the least-squares residuals are zero, so the
    // sigma^2 prior calibrates from the response standard deviation.
    let n = 10;
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = xs.iter().map(|v| 2.0 * v + 1.0).collect();
    let x = Data::new(xs, n, 1).unwrap();
    let model = fit(&config(), &x, &y, 1).unwrap();
    assert!(model.predict(&x).unwrap().iter().all(|p| p.is_finite()));
    assert!(model.sigma().iter().all(|s| s.is_finite() && *s > 0.0));
}

#[test]
fn metric_must_name_every_column_with_the_longitude_last() {
    let (x, y) = valid();
    let short = config().with_metric(vec![Metric::Euclidean]);
    assert!(matches!(
        fit(&short, &x, &y, 1).unwrap_err(),
        Error::InvalidHyperparameter { ref name, .. } if name == "metric"
    ));
    // Column 0 spans more than pi but is declared before the longitude.
    let sphere = vec![
        Metric::Spherical { sphere: 0 },
        Metric::Spherical { sphere: 0 },
    ];
    let wide = Data::from_rows(&[[-3.0, 0.1], [3.0, 0.2], [0.0, 0.3], [1.0, 0.4]]).unwrap();
    assert!(matches!(
        fit(&config().with_metric(sphere.clone()), &wide, &y, 1).unwrap_err(),
        Error::InvalidHyperparameter { ref name, .. } if name == "metric"
    ));
    let globe = Data::from_rows(&[[-1.0, -3.0], [1.0, 3.0], [0.0, 0.0], [0.5, 1.0]]).unwrap();
    let model = fit(&config().with_metric(sphere), &globe, &y, 1).unwrap();
    assert_eq!(model.predict(&globe).unwrap().len(), 4);
    // A saved model whose metric does not fit its design is rejected.
    let json = serde_json::to_string(&model).unwrap();
    let corrupt = json.replace(r#""metric":["#, r#""metric":["euclidean","#);
    assert!(serde_json::from_str::<thiessen::Fitted>(&corrupt).is_err());
}

#[test]
fn categorical_columns_take_integer_codes() {
    let codes = Data::from_rows(&[[1.0, 2.0], [2.0, 0.0], [3.0, 1.0], [4.0, 2.0]]).unwrap();
    let y = vec![0.5, 1.5, 2.5, 3.0];
    let config = config().with_metric(vec![Metric::Euclidean, Metric::Categorical]);
    let model = fit(&config, &codes, &y, 1).unwrap();
    assert_eq!(model.predict(&codes).unwrap().len(), 4);
    // A code unseen in training predicts; a non-integer does not.
    let unseen = Data::from_rows(&[[2.5, 7.0]]).unwrap();
    assert_eq!(model.predict(&unseen).unwrap().len(), 1);
    let fractional = Data::from_rows(&[[2.5, 0.5]]).unwrap();
    assert_eq!(
        model.predict(&fractional).unwrap_err(),
        Error::InvalidCategoryCode { row: 0, col: 1 }
    );
    let bad = Data::from_rows(&[[1.0, 2.0], [2.0, 0.25], [3.0, 1.0], [4.0, 2.0]]).unwrap();
    assert_eq!(
        fit(&config, &bad, &y, 1).unwrap_err(),
        Error::InvalidCategoryCode { row: 1, col: 1 }
    );
    // The levels survive a save and load; a save with levels on a
    // Euclidean column is rejected.
    let json = serde_json::to_string(&model).unwrap();
    let loaded: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(
        loaded.predict(&unseen).unwrap(),
        model.predict(&unseen).unwrap()
    );
    assert!(json.contains(r#""categories":[[],[0.0,1.0,2.0]]"#));
    let corrupt = json.replace(r#""categories":[[],"#, r#""categories":[[1.0],"#);
    assert!(serde_json::from_str::<thiessen::Fitted>(&corrupt).is_err());
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
