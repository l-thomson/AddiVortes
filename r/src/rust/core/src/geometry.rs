//! The metric of each covariate column (CRAN AddiVortes `metric` and
//! `members`): the squared distance from a row to a centre over the active
//! columns, and the centre-coordinate law of each column.

use crate::data::Data;
use crate::error::{invalid, Error, Result};
use crate::maths;
use crate::rng::{standard_normal, uniform_index, Rng};

/// The metric of one covariate column,
/// [`GeometryParams::metric`](crate::GeometryParams::metric).
/// Columns of different metrics combine additively: the key of a row
/// against a centre is the sum of the squared distances of its metrics
/// over the active columns.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(not(feature = "experimental"), derive(Eq))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Metric {
    /// Squared Euclidean distance on the column scaled to [-0.5, 0.5] over
    /// its training range; centre coordinates N(0, sigma_c^2). The
    /// default for every column.
    Euclidean,
    /// One coordinate, in radians, of the sphere labelled `sphere`. The
    /// columns declared for a sphere are its latitudes and, last, its
    /// longitude, whose period is 2 pi; a sphere of one column is a
    /// circle. The distance is the squared great-circle angle, computed
    /// with the row's own coordinates in the columns a tessellation does
    /// not use. The column is not scaled; centre coordinates are
    /// N(mid, sd^2) with mid the column's training mid-range and sd its
    /// range over 2 Phi^-1(0.75), the longitude wrapped to [-pi, pi]
    /// (CRAN AddiVortes `metric = "S"`, `members`).
    Spherical {
        /// Sphere label; columns sharing a label form one sphere.
        sphere: usize,
    },
    /// Integer level codes of a categorical covariate; the levels are the
    /// distinct training values. A mismatch between the row and the
    /// centre contributes 2 / n^2, n the number of levels (the Eskin et
    /// al. 2002 weight; CRAN AddiVortes `metric = "C"` with
    /// `cat.onehot = FALSE`). The column is not scaled; centre
    /// coordinates are uniform over the levels. A non-integer value is
    /// rejected at fit and predict.
    Categorical,
    /// Minkowski distance of order `p` >= 1 on the column scaled to
    /// [-0.5, 0.5] over its training range; centre coordinates
    /// N(0, sigma_c^2), as Euclidean. The active Minkowski columns of one
    /// order form a group contributing (sum_d |x_d - c_d|^p)^(2 / p) to
    /// the key, so p = 2 reproduces Euclidean exactly. The centre prior
    /// and the structural moves are those of Euclidean; only the
    /// assignment of rows to cells changes. Experimental
    /// (`docs/experimental.md`).
    #[cfg(feature = "experimental")]
    Minkowski {
        /// The order, at least 1; 1 is the Manhattan distance.
        p: f64,
    },
    /// [`Metric::Minkowski`] of order 1 under its usual name.
    /// Experimental (`docs/experimental.md`).
    #[cfg(feature = "experimental")]
    Manhattan,
}

/// The column structure of a fit: the metric of each column, the columns
/// of each sphere, and the levels of each categorical column.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Geometry {
    kinds: Vec<Metric>,
    /// Columns of each sphere in declaration order, the last the longitude.
    spheres: Vec<Vec<usize>>,
    /// Sphere index of each spherical column.
    sphere_of: Vec<Option<usize>>,
    /// Sorted distinct training values of each categorical column; empty
    /// for the other columns.
    categories: Vec<Vec<f64>>,
    /// 2 / n^2 of each categorical column; 0 for the other columns.
    weights: Vec<f64>,
    /// Minkowski column groups, (order, columns), one per distinct order.
    #[cfg(feature = "experimental")]
    minkowski: Vec<(f64, Vec<usize>)>,
}

impl Geometry {
    /// p Euclidean columns.
    pub(crate) fn euclidean(p: usize) -> Self {
        Self {
            kinds: vec![Metric::Euclidean; p],
            spheres: Vec::new(),
            sphere_of: vec![None; p],
            categories: vec![Vec::new(); p],
            weights: vec![0.0; p],
            #[cfg(feature = "experimental")]
            minkowski: Vec::new(),
        }
    }

    /// The geometry of `metric` over p columns with the categorical levels
    /// still to be learnt; an empty `metric` is p Euclidean columns.
    ///
    /// # Errors
    ///
    /// `InvalidHyperparameter` for `metric` when it is neither empty nor
    /// of length p.
    pub(crate) fn structure(metric: &[Metric], p: usize) -> Result<Self> {
        if metric.is_empty() {
            return Ok(Self::euclidean(p));
        }
        if metric.len() != p {
            return Err(invalid(
                "metric",
                format!(
                    "must name every column: {} entries for p = {p} columns",
                    metric.len()
                ),
            ));
        }
        let mut labels: Vec<usize> = Vec::new();
        let mut spheres: Vec<Vec<usize>> = Vec::new();
        let mut sphere_of = vec![None; p];
        for (col, kind) in metric.iter().enumerate() {
            if let Metric::Spherical { sphere } = *kind {
                let index = match labels.iter().position(|&l| l == sphere) {
                    Some(index) => index,
                    None => {
                        labels.push(sphere);
                        spheres.push(Vec::new());
                        spheres.len() - 1
                    }
                };
                spheres[index].push(col);
                sphere_of[col] = Some(index);
            }
        }
        #[cfg(feature = "experimental")]
        let minkowski = {
            let mut groups: Vec<(f64, Vec<usize>)> = Vec::new();
            for (col, kind) in metric.iter().enumerate() {
                let p = match *kind {
                    Metric::Minkowski { p } => p,
                    Metric::Manhattan => 1.0,
                    _ => continue,
                };
                match groups.iter_mut().find(|(q, _)| q.to_bits() == p.to_bits()) {
                    Some((_, cols)) => cols.push(col),
                    None => groups.push((p, vec![col])),
                }
            }
            groups
        };
        Ok(Self {
            kinds: metric.to_vec(),
            spheres,
            sphere_of,
            categories: vec![Vec::new(); p],
            weights: vec![0.0; p],
            #[cfg(feature = "experimental")]
            minkowski,
        })
    }

    /// The geometry of `metric` over the training design `x`, learning
    /// the levels of the categorical columns.
    ///
    /// # Errors
    ///
    /// [`Geometry::structure`]; `InvalidCategoryCode` for a non-integer
    /// value in a categorical column.
    pub(crate) fn fit(metric: &[Metric], x: &Data) -> Result<Self> {
        let p = x.n_cols();
        let mut geometry = Self::structure(metric, p)?;
        geometry.check_codes(x)?;
        for col in 0..p {
            if geometry.kinds[col] == Metric::Categorical {
                let mut levels: Vec<f64> = (0..x.n_rows()).map(|r| x.row(r)[col]).collect();
                levels.sort_by(f64::total_cmp);
                levels.dedup();
                geometry.set_categories(col, levels);
            }
        }
        Ok(geometry)
    }

    /// The geometry of a fitted model from its `metric` and the stored
    /// levels of its categorical columns (empty for the others).
    ///
    /// # Errors
    ///
    /// [`Geometry::structure`]; `InvalidSavedModel` when the levels do
    /// not match the metric.
    pub(crate) fn with_categories(
        metric: &[Metric],
        p: usize,
        categories: &[Vec<f64>],
    ) -> Result<Self> {
        let mut geometry = Self::structure(metric, p)?;
        let bad = |reason: &str| Error::InvalidSavedModel {
            reason: reason.into(),
        };
        if categories.len() != p {
            return Err(bad("categorical levels must be stored for every column"));
        }
        for (col, levels) in categories.iter().enumerate() {
            match geometry.kinds[col] {
                Metric::Categorical => {
                    if levels.is_empty()
                        || levels.iter().any(|v| !v.is_finite() || v.fract() != 0.0)
                        || levels.windows(2).any(|w| w[0] >= w[1])
                    {
                        return Err(bad(
                            "a categorical column needs sorted, distinct integer levels",
                        ));
                    }
                    geometry.set_categories(col, levels.clone());
                }
                _ if !levels.is_empty() => {
                    return Err(bad("only categorical columns carry levels"));
                }
                _ => {}
            }
        }
        Ok(geometry)
    }

    fn set_categories(&mut self, col: usize, levels: Vec<f64>) {
        let n = levels.len() as f64;
        self.weights[col] = 2.0 / (n * n);
        self.categories[col] = levels;
    }

    /// The levels of each column, as [`Geometry::with_categories`] takes
    /// them.
    pub(crate) fn categories(&self) -> &[Vec<f64>] {
        &self.categories
    }

    /// Whether column `col` is min-max scaled by the [`Scaler`](crate::Scaler).
    pub(crate) fn scaled(&self, col: usize) -> bool {
        match self.kinds[col] {
            Metric::Euclidean => true,
            #[cfg(feature = "experimental")]
            Metric::Minkowski { .. } | Metric::Manhattan => true,
            _ => false,
        }
    }

    /// Every categorical value of `x` is an integer.
    ///
    /// # Errors
    ///
    /// `InvalidCategoryCode` at the first offence, row-major.
    pub(crate) fn check_codes(&self, x: &Data) -> Result<()> {
        let categorical: Vec<usize> = (0..x.n_cols())
            .filter(|&col| self.kinds[col] == Metric::Categorical)
            .collect();
        if categorical.is_empty() {
            return Ok(());
        }
        for row in 0..x.n_rows() {
            let values = x.row(row);
            for &col in &categorical {
                if values[col].fract() != 0.0 {
                    return Err(Error::InvalidCategoryCode { row, col });
                }
            }
        }
        Ok(())
    }

    /// Squared distance from `row` (a full p-length row) to a centre with
    /// coordinates `centre` on the columns `dims`, in `dims` order.
    pub(crate) fn key(&self, row: &[f64], dims: &[usize], centre: &[f64]) -> f64 {
        let mut key = 0.0;
        let plain = self.spheres.is_empty() && self.weights.iter().all(|&w| w == 0.0);
        #[cfg(feature = "experimental")]
        let plain = plain && self.minkowski.is_empty();
        if plain {
            for (&dim, &c) in dims.iter().zip(centre) {
                let diff = row[dim] - c;
                key += diff * diff;
            }
            return key;
        }
        for (&dim, &c) in dims.iter().zip(centre) {
            match self.kinds[dim] {
                Metric::Euclidean => {
                    let diff = row[dim] - c;
                    key += diff * diff;
                }
                Metric::Categorical => {
                    if row[dim] != c {
                        key += self.weights[dim];
                    }
                }
                Metric::Spherical { .. } => {}
                #[cfg(feature = "experimental")]
                Metric::Minkowski { .. } | Metric::Manhattan => {}
            }
        }
        #[cfg(feature = "experimental")]
        for (p, cols) in &self.minkowski {
            let mut sum = 0.0;
            for (&dim, &c) in dims.iter().zip(centre) {
                if !cols.contains(&dim) {
                    continue;
                }
                let diff = (row[dim] - c).abs();
                sum += if *p == 2.0 {
                    diff * diff
                } else {
                    maths::powf(diff, *p)
                };
            }
            key += if *p == 2.0 {
                sum
            } else if *p == 1.0 {
                sum * sum
            } else {
                maths::powf(sum, 2.0 / *p)
            };
        }
        for (s, cols) in self.spheres.iter().enumerate() {
            if dims.iter().all(|&dim| self.sphere_of[dim] != Some(s)) {
                continue;
            }
            let coordinate = |col: usize| -> (f64, f64) {
                let a = row[col];
                let b = match dims.iter().position(|&dim| dim == col) {
                    Some(j) => centre[j],
                    None => a,
                };
                (a, b)
            };
            key += sphere_angle_sq(cols, &coordinate);
        }
        key
    }

    /// The centre-coordinate law of every column from the training design
    /// `x`: N(0, sigma_c^2) on Euclidean columns; on spherical columns
    /// N(mid, sd^2) from the column's range, wrapped for a longitude;
    /// uniform over the levels on categorical columns.
    ///
    /// # Errors
    ///
    /// `InvalidHyperparameter` for `metric` when a latitude column (not
    /// the last of its sphere) spans more than pi.
    pub(crate) fn laws(&self, x: &Data, sigma_c: f64) -> Result<Vec<CoordinateLaw>> {
        let n = x.n_rows();
        let p = x.n_cols();
        let q75 = maths::normal_quantile(0.75);
        (0..p)
            .map(|col| match self.kinds[col] {
                Metric::Euclidean => Ok(CoordinateLaw::Normal {
                    mean: 0.0,
                    sd: sigma_c,
                }),
                #[cfg(feature = "experimental")]
                Metric::Minkowski { .. } | Metric::Manhattan => Ok(CoordinateLaw::Normal {
                    mean: 0.0,
                    sd: sigma_c,
                }),
                Metric::Categorical => Ok(CoordinateLaw::Uniform {
                    levels: self.categories[col].clone(),
                }),
                Metric::Spherical { .. } => {
                    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
                    for r in 0..n {
                        let v = x.values()[r * p + col];
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                    let sphere = self.sphere_of[col].expect("spherical column");
                    let last = *self.spheres[sphere].last().expect("non-empty sphere") == col;
                    if !last && hi - lo > std::f64::consts::PI {
                        return Err(invalid(
                            "metric",
                            format!(
                                "column {col} spans more than pi but is not the last column of \
                                 its sphere; declare the longitude last"
                            ),
                        ));
                    }
                    let mean = 0.5 * (lo + hi);
                    let sd = 0.5 * (hi - lo) / q75;
                    Ok(if last {
                        CoordinateLaw::WrappedNormal { mean, sd }
                    } else {
                        CoordinateLaw::Normal { mean, sd }
                    })
                }
            })
            .collect()
    }
}

/// Squared great-circle angle between two points of a sphere whose
/// coordinates `coordinate(col)` returns as (point, centre) pairs, `cols`
/// ending with the longitude: on a circle the shorter arc; otherwise the
/// nested spherical law of cosines, cos c = sin a sin b + cos a cos b cos C,
/// evaluated from the longitude inwards (CRAN AddiVortes
/// `spherical_distance`).
fn sphere_angle_sq(cols: &[usize], coordinate: &dyn Fn(usize) -> (f64, f64)) -> f64 {
    let k = cols.len();
    let (a, b) = coordinate(cols[k - 1]);
    if k == 1 {
        let arc = (a - b).abs();
        let other = 2.0 * std::f64::consts::PI - arc;
        let shorter = if arc < other { arc } else { other };
        return shorter * shorter;
    }
    let mut angle = maths::cos(a - b);
    for i in (0..k - 1).rev() {
        let (a, b) = coordinate(cols[i]);
        let internal = (maths::sin(a) * maths::sin(b) + maths::cos(a) * maths::cos(b) * angle)
            .clamp(-1.0, 1.0);
        angle = if i == 0 {
            maths::acos(internal)
        } else {
            internal
        };
    }
    angle * angle
}

/// The prior and proposal law of one centre coordinate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CoordinateLaw {
    /// N(mean, sd^2).
    Normal { mean: f64, sd: f64 },
    /// N(mean, sd^2) wrapped to [-pi, pi].
    WrappedNormal { mean: f64, sd: f64 },
    /// Uniform over `levels`.
    Uniform { levels: Vec<f64> },
}

impl CoordinateLaw {
    pub(crate) fn draw(&self, rng: &mut Rng) -> f64 {
        match self {
            CoordinateLaw::Normal { mean, sd } => mean + sd * standard_normal(rng),
            CoordinateLaw::WrappedNormal { mean, sd } => {
                let mut v = mean + sd * standard_normal(rng);
                while v > std::f64::consts::PI {
                    v -= 2.0 * std::f64::consts::PI;
                }
                while v < -std::f64::consts::PI {
                    v += 2.0 * std::f64::consts::PI;
                }
                v
            }
            CoordinateLaw::Uniform { levels } => levels[uniform_index(levels.len(), rng)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::chain_rng;
    use std::f64::consts::PI;

    fn sphere2() -> Geometry {
        Geometry::structure(
            &[
                Metric::Spherical { sphere: 0 },
                Metric::Spherical { sphere: 0 },
            ],
            2,
        )
        .unwrap()
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    #[test]
    fn construction_and_validation() {
        let g = Geometry::structure(&[], 3).unwrap();
        assert_eq!(g, Geometry::euclidean(3));
        assert!(Geometry::structure(&[Metric::Euclidean], 2).is_err());
        let g = Geometry::structure(
            &[
                Metric::Spherical { sphere: 7 },
                Metric::Euclidean,
                Metric::Spherical { sphere: 7 },
                Metric::Spherical { sphere: 2 },
            ],
            4,
        )
        .unwrap();
        assert_eq!(g.spheres, vec![vec![0, 2], vec![3]]);
        assert_eq!(g.sphere_of, vec![Some(0), None, Some(0), Some(1)]);
        assert!(g.scaled(1));
        assert!(!g.scaled(0));
    }

    #[test]
    fn euclidean_key_is_the_squared_distance_over_the_active_columns() {
        let g = Geometry::euclidean(3);
        close(g.key(&[1.0, 2.0, 3.0], &[2, 0], &[0.0, 0.5]), 9.0 + 0.25);
    }

    #[test]
    fn great_circle_hand_values() {
        let g = sphere2();
        // Equator, a quarter turn apart: pi / 2.
        close(
            g.key(&[0.0, 0.0], &[0, 1], &[0.0, PI / 2.0]),
            (PI / 2.0).powi(2),
        );
        // Pole to equator: pi / 2, whatever the longitude.
        close(
            g.key(&[PI / 2.0, 1.0], &[0, 1], &[0.0, -2.0]),
            (PI / 2.0).powi(2),
        );
        // Across the date line: the short way round.
        close(
            g.key(&[0.0, PI - 0.1], &[0, 1], &[0.0, -PI + 0.1]),
            0.2_f64.powi(2),
        );
        // Only the latitude active: the centre takes the row's longitude.
        close(g.key(&[0.3, 2.0], &[0], &[-0.2]), 0.5_f64.powi(2));
        // Only the longitude active: the arc on the row's parallel.
        let row = [0.6, 0.0];
        let expected = maths::acos(
            maths::sin(0.6) * maths::sin(0.6) + maths::cos(0.6) * maths::cos(0.6) * maths::cos(1.0),
        );
        close(g.key(&row, &[1], &[1.0]), expected * expected);
    }

    #[test]
    fn circle_takes_the_shorter_arc() {
        let g = Geometry::structure(&[Metric::Spherical { sphere: 0 }], 1).unwrap();
        close(g.key(&[PI - 0.25], &[0], &[-PI + 0.25]), 0.25);
        close(g.key(&[0.0], &[0], &[1.0]), 1.0);
    }

    #[test]
    fn mixed_columns_add() {
        let g = Geometry::structure(
            &[
                Metric::Euclidean,
                Metric::Spherical { sphere: 0 },
                Metric::Spherical { sphere: 0 },
            ],
            3,
        )
        .unwrap();
        close(
            g.key(&[0.5, 0.0, 0.0], &[0, 1, 2], &[0.0, 0.0, 1.0]),
            0.25 + 1.0,
        );
    }

    #[test]
    fn categorical_levels_weights_and_codes() {
        let metric = [Metric::Euclidean, Metric::Categorical, Metric::Categorical];
        let x = Data::from_rows(&[
            [0.1, 3.0, 1.0],
            [0.2, 1.0, 2.0],
            [0.3, 3.0, 1.0],
            [0.4, 2.0, 1.0],
        ])
        .unwrap();
        let g = Geometry::fit(&metric, &x).unwrap();
        assert_eq!(
            g.categories(),
            &[vec![], vec![1.0, 2.0, 3.0], vec![1.0, 2.0]]
        );
        assert!(!g.scaled(1));
        // Column 1 mismatch 2 / 9, column 2 mismatch 2 / 4, Euclidean part.
        close(
            g.key(&[0.5, 3.0, 1.0], &[0, 1, 2], &[0.0, 1.0, 1.0]),
            0.25 + 2.0 / 9.0,
        );
        close(g.key(&[0.5, 3.0, 1.0], &[2, 1], &[2.0, 3.0]), 0.5);
        close(g.key(&[0.5, 3.0, 1.0], &[1], &[3.0]), 0.0);
        // An unseen code at predict is a mismatch against every centre.
        close(g.key(&[0.5, 7.0, 1.0], &[1], &[3.0]), 2.0 / 9.0);
        let bad = Data::from_rows(&[[0.1, 1.5, 1.0], [0.2, 1.0, 2.0]]).unwrap();
        assert_eq!(
            Geometry::fit(&metric, &bad).unwrap_err(),
            Error::InvalidCategoryCode { row: 0, col: 1 }
        );
        assert_eq!(
            g.check_codes(&bad).unwrap_err(),
            Error::InvalidCategoryCode { row: 0, col: 1 }
        );
        // The stored levels rebuild the geometry; inconsistent levels do not.
        assert_eq!(
            Geometry::with_categories(&metric, 3, g.categories()).unwrap(),
            g
        );
        assert!(Geometry::with_categories(&metric, 3, &[vec![], vec![], vec![1.0, 2.0]]).is_err());
        assert!(Geometry::with_categories(&metric, 3, &[vec![1.0], vec![1.0], vec![1.0]]).is_err());
        assert!(
            Geometry::with_categories(&metric, 3, &[vec![], vec![2.0, 1.0], vec![1.0]]).is_err()
        );
        let mut rng = chain_rng(5);
        let laws = g.laws(&x, 0.8).unwrap();
        assert_eq!(
            laws[1],
            CoordinateLaw::Uniform {
                levels: vec![1.0, 2.0, 3.0]
            }
        );
        for _ in 0..100 {
            assert!([1.0, 2.0, 3.0].contains(&laws[1].draw(&mut rng)));
        }
    }

    #[test]
    fn laws_follow_the_training_range_and_wrap_the_longitude() {
        let g = sphere2();
        let x = Data::from_rows(&[[-0.5, -3.0], [0.5, 3.0], [0.0, 0.0]]).unwrap();
        let laws = g.laws(&x, 0.8).unwrap();
        let sd = |range: f64| 0.5 * range / maths::normal_quantile(0.75);
        assert_eq!(
            laws[0],
            CoordinateLaw::Normal {
                mean: 0.0,
                sd: sd(1.0)
            }
        );
        assert_eq!(
            laws[1],
            CoordinateLaw::WrappedNormal {
                mean: 0.0,
                sd: sd(6.0)
            }
        );
        let mut rng = chain_rng(3);
        for _ in 0..1000 {
            let v = laws[1].draw(&mut rng);
            assert!((-PI..=PI).contains(&v));
        }
        assert_eq!(
            Geometry::euclidean(2).laws(&x, 0.8).unwrap()[1],
            CoordinateLaw::Normal { mean: 0.0, sd: 0.8 }
        );
        // A latitude spanning more than pi is rejected.
        let wide = Data::from_rows(&[[-3.0, 0.0], [3.0, 1.0]]).unwrap();
        assert!(g.laws(&wide, 0.8).is_err());
    }

    #[cfg(feature = "experimental")]
    mod minkowski {
        use super::*;

        #[test]
        fn order_two_matches_euclidean_bit_for_bit() {
            let g = Geometry::structure(&[Metric::Minkowski { p: 2.0 }; 3], 3).unwrap();
            let e = Geometry::euclidean(3);
            let row = [0.31, -0.47, 0.055];
            let centre = [-0.12, 0.4];
            let dims = [2, 0];
            assert_eq!(
                g.key(&row, &dims, &centre).to_bits(),
                e.key(&row, &dims, &centre).to_bits()
            );
        }

        #[test]
        fn manhattan_equals_order_one() {
            let m = Geometry::structure(&[Metric::Manhattan; 2], 2).unwrap();
            let o = Geometry::structure(&[Metric::Minkowski { p: 1.0 }; 2], 2).unwrap();
            let row = [0.2, -0.3];
            let key = m.key(&row, &[0, 1], &[-0.1, 0.25]);
            assert_eq!(key, o.key(&row, &[0, 1], &[-0.1, 0.25]));
            close(key, (0.3_f64 + 0.55).powi(2));
        }

        #[test]
        fn a_group_adds_to_the_euclidean_part() {
            let g = Geometry::structure(
                &[
                    Metric::Euclidean,
                    Metric::Minkowski { p: 3.0 },
                    Metric::Minkowski { p: 3.0 },
                ],
                3,
            )
            .unwrap();
            let key = g.key(&[0.5, 0.2, -0.1], &[0, 1, 2], &[0.1, -0.2, 0.3]);
            let group: f64 = 0.4_f64.powi(3) + 0.4_f64.powi(3);
            close(key, 0.16 + group.powf(2.0 / 3.0));
        }

        #[test]
        fn distinct_orders_are_separate_groups() {
            let g = Geometry::structure(
                &[Metric::Minkowski { p: 1.0 }, Metric::Minkowski { p: 3.0 }],
                2,
            )
            .unwrap();
            let key = g.key(&[0.5, 0.2], &[0, 1], &[0.1, -0.2]);
            close(key, 0.16 + 0.4_f64.powi(3).powf(2.0 / 3.0));
        }

        #[test]
        fn an_inactive_group_contributes_nothing() {
            let g = Geometry::structure(&[Metric::Euclidean, Metric::Manhattan], 2).unwrap();
            close(g.key(&[0.5, 0.2], &[0], &[0.1]), 0.16);
        }

        #[test]
        fn scaled_and_law_follow_euclidean() {
            let g =
                Geometry::structure(&[Metric::Minkowski { p: 1.5 }, Metric::Manhattan], 2).unwrap();
            assert!(g.scaled(0));
            assert!(g.scaled(1));
            let x = Data::from_rows(&[[0.0, 1.0], [1.0, 2.0]]).unwrap();
            for law in g.laws(&x, 0.8).unwrap() {
                assert_eq!(law, CoordinateLaw::Normal { mean: 0.0, sd: 0.8 });
            }
        }
    }

    mod props {
        use super::*;
        use proptest::prelude::*;

        fn point() -> impl Strategy<Value = (f64, f64, f64, i8)> {
            (-PI / 2.0..PI / 2.0, -PI..PI, -1.0..1.0_f64, 0..4_i8)
        }

        // The key is a squared distance: its square root is a metric on
        // the sphere, the Euclidean part, the categorical part and their
        // sum.
        proptest! {
            #[test]
            fn metric_axioms((a, b, c) in (point(), point(), point())) {
                let metric = [
                    Metric::Euclidean,
                    Metric::Spherical { sphere: 0 },
                    Metric::Spherical { sphere: 0 },
                    Metric::Categorical,
                ];
                let levels = [vec![], vec![], vec![], vec![0.0, 1.0, 2.0, 3.0]];
                let g = Geometry::with_categories(&metric, 4, &levels).unwrap();
                let dims = [0, 1, 2, 3];
                let d = |u: (f64, f64, f64, i8), v: (f64, f64, f64, i8)| {
                    g.key(
                        &[u.2, u.0, u.1, f64::from(u.3)],
                        &dims,
                        &[v.2, v.0, v.1, f64::from(v.3)],
                    )
                    .sqrt()
                };
                prop_assert!(d(a, b) >= 0.0);
                prop_assert!(d(a, a) < 1e-7);
                prop_assert!((d(a, b) - d(b, a)).abs() < 1e-9);
                prop_assert!(d(a, c) <= d(a, b) + d(b, c) + 1e-9);
            }
        }

        // The Minkowski key is a squared distance for every order p >= 1.
        #[cfg(feature = "experimental")]
        proptest! {
            #[test]
            fn minkowski_axioms(
                a in (-1.0..1.0_f64, -1.0..1.0_f64, -1.0..1.0_f64),
                b in (-1.0..1.0_f64, -1.0..1.0_f64, -1.0..1.0_f64),
                c in (-1.0..1.0_f64, -1.0..1.0_f64, -1.0..1.0_f64),
                p in 1.0..6.0_f64,
            ) {
                let g = Geometry::structure(&[Metric::Minkowski { p }; 3], 3).unwrap();
                let dims = [0, 1, 2];
                let d = |u: (f64, f64, f64), v: (f64, f64, f64)| {
                    g.key(&[u.0, u.1, u.2], &dims, &[v.0, v.1, v.2]).sqrt()
                };
                prop_assert!(d(a, b) >= 0.0);
                prop_assert!(d(a, a) == 0.0);
                prop_assert!((d(a, b) - d(b, a)).abs() < 1e-12);
                prop_assert!(d(a, c) <= d(a, b) + d(b, c) + 1e-9);
            }
        }
    }
}
