//! A Voronoi tessellation of a covariate subspace and the assignment of
//! observations to its cells.

use crate::data::Data;
use crate::error::{Error, Result};
use crate::geometry::Geometry;

/// One tessellation: b centres in a d-dimensional subspace of the scaled
/// covariate space, one cell mean per centre.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "TessellationParts")]
pub struct Tessellation {
    /// Row-major b by d centre coordinates (scaled space).
    pub(crate) centres: Vec<f64>,
    /// The d active covariates, zero-based column indices, in centre
    /// coordinate order.
    pub(crate) dims: Vec<usize>,
    /// Cell means mu_1..mu_b (scaled space); under the linear cell basis
    /// the intercepts, the value at the cell's centre.
    pub(crate) mus: Vec<f64>,
    /// Row-major b by d slopes of the linear cell basis (scaled space);
    /// empty under the constant basis.
    pub(crate) betas: Vec<f64>,
    /// The soft-membership kernel bandwidth (scaled space); `None` under
    /// hard membership. Experimental (`docs/experimental.md`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tau: Option<f64>,
}

impl Tessellation {
    /// Number of cells b.
    pub fn n_cells(&self) -> usize {
        self.mus.len()
    }

    /// Number of active covariates d.
    pub fn n_dims(&self) -> usize {
        self.dims.len()
    }

    /// The active covariates, zero-based column indices.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Cell means, scaled space.
    pub fn mus(&self) -> &[f64] {
        &self.mus
    }

    /// Centre `k`'s coordinates, one per active covariate.
    pub fn centre(&self, k: usize) -> &[f64] {
        let d = self.dims.len();
        &self.centres[k * d..(k + 1) * d]
    }

    /// Squared distance under `geometry` from `row` (a full p-length scaled
    /// row) to centre `k` over the active covariates.
    pub(crate) fn key(&self, row: &[f64], k: usize, geometry: &Geometry) -> f64 {
        geometry.key(row, &self.dims, self.centre(k))
    }

    /// The nearest centre to `row` and its key; ties go to the lowest index.
    pub(crate) fn nearest(&self, row: &[f64], geometry: &Geometry) -> (usize, f64) {
        let mut best = f64::INFINITY;
        let mut best_cell = 0;
        for k in 0..self.n_cells() {
            let key = self.key(row, k, geometry);
            if key < best {
                best = key;
                best_cell = k;
            }
        }
        (best_cell, best)
    }

    /// The soft-membership kernel bandwidth; `None` under hard
    /// membership.
    #[cfg(feature = "experimental")]
    pub fn bandwidth(&self) -> Option<f64> {
        self.tau
    }

    /// The value of the tessellation at `row`: the mean of the cell `row`
    /// falls in, tilted by the cell's slopes under the linear basis;
    /// under soft membership the kernel-weighted sum of the cell means.
    pub(crate) fn value_at(&self, row: &[f64], geometry: &Geometry) -> f64 {
        if let Some(tau) = self.tau {
            return self.soft_value_at(row, geometry, tau);
        }
        self.value_in_cell(self.nearest(row, geometry).0, row)
    }

    /// The kernel-weighted sum of the cell means at `row`: weights
    /// proportional to exp(-key / (2 tau^2)) over the centres, computed
    /// from the smallest key so the nearest centre's factor is 1.
    fn soft_value_at(&self, row: &[f64], geometry: &Geometry, tau: f64) -> f64 {
        let b = self.n_cells();
        let mut keys = Vec::with_capacity(b);
        let mut min = f64::INFINITY;
        for k in 0..b {
            let key = self.key(row, k, geometry);
            min = min.min(key);
            keys.push(key);
        }
        let scale = 1.0 / (2.0 * tau * tau);
        let mut total = 0.0;
        let mut value = 0.0;
        for (key, &mu) in keys.iter().zip(&self.mus) {
            let g = crate::maths::exp(-(key - min) * scale);
            total += g;
            value += g * mu;
        }
        value / total
    }

    /// The value cell `cell` contributes at `row`: the cell mean, plus,
    /// under the linear basis, the slopes against the row's offset from
    /// the cell's centre on the active covariates.
    pub(crate) fn value_in_cell(&self, cell: usize, row: &[f64]) -> f64 {
        let mut value = self.mus[cell];
        if !self.betas.is_empty() {
            let d = self.dims.len();
            let centre = self.centre(cell);
            for (j, &dim) in self.dims.iter().enumerate() {
                value += self.betas[cell * d + j] * (row[dim] - centre[j]);
            }
        }
        value
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TessellationParts {
    centres: Vec<f64>,
    dims: Vec<usize>,
    mus: Vec<f64>,
    #[serde(default)]
    betas: Vec<f64>,
    #[serde(default)]
    tau: Option<f64>,
}

impl TryFrom<TessellationParts> for Tessellation {
    type Error = Error;

    fn try_from(parts: TessellationParts) -> Result<Self> {
        let bad = |reason: &str| Error::InvalidSavedModel {
            reason: reason.into(),
        };
        let (b, d) = (parts.mus.len(), parts.dims.len());
        if b == 0 || d == 0 {
            return Err(bad(
                "a tessellation needs at least one cell and one dimension",
            ));
        }
        if parts.centres.len() != b * d {
            return Err(bad(
                "centre buffer length does not match cells by dimensions",
            ));
        }
        if !parts.betas.is_empty() && parts.betas.len() != b * d {
            return Err(bad(
                "slope buffer length does not match cells by dimensions",
            ));
        }
        if parts
            .centres
            .iter()
            .chain(&parts.mus)
            .chain(&parts.betas)
            .any(|v| !v.is_finite())
        {
            return Err(bad("tessellation values must be finite"));
        }
        for (i, dim) in parts.dims.iter().enumerate() {
            if parts.dims[..i].contains(dim) {
                return Err(bad("tessellation dimensions must be distinct"));
            }
        }
        if let Some(tau) = parts.tau {
            #[cfg(not(feature = "experimental"))]
            {
                let _ = tau;
                return Err(bad(
                    "the soft-membership bandwidth needs the experimental feature",
                ));
            }
            #[cfg(feature = "experimental")]
            if !(tau.is_finite() && tau > 0.0) {
                return Err(bad(
                    "the soft-membership bandwidth must be finite and positive",
                ));
            }
        }
        Ok(Self {
            centres: parts.centres,
            dims: parts.dims,
            mus: parts.mus,
            betas: parts.betas,
            tau: parts.tau,
        })
    }
}

/// The structural change a proposal makes relative to the tessellation it
/// was proposed from, so the cached assignment can be updated rather than
/// recomputed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Delta {
    /// One centre appended at the highest index; dims unchanged.
    CentreAdded,
    /// One centre removed; higher indices shift down by one; dims unchanged.
    CentreRemoved(usize),
    /// One centre's coordinates changed in place; dims unchanged.
    CentreMoved(usize),
    /// dims changed: every key is stale.
    Full,
}

/// Cached nearest-centre assignment of every observation, with the winning
/// key, for one tessellation; under soft membership also every key.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Assignment {
    /// Cell index per observation.
    pub(crate) cells: Vec<usize>,
    /// Squared distance to the assigned centre per observation.
    keys: Vec<f64>,
    /// Column-major b by n keys of every observation against every
    /// centre; `None` under hard membership, whose moves need only the
    /// winner.
    soft: Option<Vec<f64>>,
}

impl Assignment {
    /// Assignment of every row of `x` under `t`, computed in full.
    pub(crate) fn full(x: &Data, t: &Tessellation, geometry: &Geometry) -> Self {
        let n = x.n_rows();
        let mut cells = Vec::with_capacity(n);
        let mut keys = Vec::with_capacity(n);
        if t.tau.is_none() {
            for i in 0..n {
                let (cell, key) = t.nearest(x.row(i), geometry);
                cells.push(cell);
                keys.push(key);
            }
            return Self {
                cells,
                keys,
                soft: None,
            };
        }
        let b = t.n_cells();
        let mut soft = vec![0.0; b * n];
        for i in 0..n {
            let row = x.row(i);
            let mut best = f64::INFINITY;
            let mut best_cell = 0;
            for k in 0..b {
                let key = t.key(row, k, geometry);
                soft[k * n + i] = key;
                if key < best {
                    best = key;
                    best_cell = k;
                }
            }
            cells.push(best_cell);
            keys.push(best);
        }
        Self {
            cells,
            keys,
            soft: Some(soft),
        }
    }

    /// Row-major n by b kernel weights at bandwidth `tau`: per
    /// observation exp(-(key - winning key) / (2 tau^2)) over the
    /// centres, normalised, so the nearest centre's factor is 1.
    pub(crate) fn soft_weights(&self, tau: f64) -> Vec<f64> {
        let keys = self.soft.as_ref().expect("soft keys are cached");
        let n = self.cells.len();
        let b = keys.len() / n;
        let scale = 1.0 / (2.0 * tau * tau);
        let mut weights = vec![0.0; n * b];
        for i in 0..n {
            let min = self.keys[i];
            let mut total = 0.0;
            for k in 0..b {
                let g = crate::maths::exp(-(keys[k * n + i] - min) * scale);
                weights[i * b + k] = g;
                total += g;
            }
            for w in &mut weights[i * b..(i + 1) * b] {
                *w /= total;
            }
        }
        weights
    }

    /// Assignment under `new`, which differs from the tessellation this
    /// cache was built for by `delta`. Equal to [`Assignment::full`] on the
    /// same inputs. With dims unchanged an untouched centre's key against
    /// any observation is unchanged, so only pairs involving the touched
    /// centre are recomputed.
    pub(crate) fn updated(
        &self,
        x: &Data,
        new: &Tessellation,
        delta: Delta,
        geometry: &Geometry,
    ) -> Self {
        let n = x.n_rows();
        match delta {
            Delta::Full => Self::full(x, new, geometry),
            Delta::CentreAdded => {
                let added = new.n_cells() - 1;
                let mut out = self.clone();
                if let Some(soft) = &mut out.soft {
                    soft.reserve(n);
                }
                for i in 0..n {
                    let key = new.key(x.row(i), added, geometry);
                    if let Some(soft) = &mut out.soft {
                        soft.push(key);
                    }
                    if key < out.keys[i] {
                        out.keys[i] = key;
                        out.cells[i] = added;
                    }
                }
                out
            }
            Delta::CentreMoved(moved) => {
                let mut out = self.clone();
                for i in 0..n {
                    let row = x.row(i);
                    let key = new.key(row, moved, geometry);
                    if let Some(soft) = &mut out.soft {
                        soft[moved * n + i] = key;
                    }
                    if self.cells[i] == moved {
                        let (cell, key) = new.nearest(row, geometry);
                        out.cells[i] = cell;
                        out.keys[i] = key;
                    } else {
                        // Strict comparison keeps the lowest-index tie rule:
                        // `moved` wins only when nearer than the incumbent
                        // or equal with a lower index.
                        if key < self.keys[i] || (key == self.keys[i] && moved < self.cells[i]) {
                            out.cells[i] = moved;
                            out.keys[i] = key;
                        }
                    }
                }
                out
            }
            Delta::CentreRemoved(removed) => {
                let mut out = self.clone();
                if let Some(soft) = &mut out.soft {
                    soft.drain(removed * n..(removed + 1) * n);
                }
                for i in 0..n {
                    if self.cells[i] == removed {
                        let (cell, key) = new.nearest(x.row(i), geometry);
                        out.cells[i] = cell;
                        out.keys[i] = key;
                    } else if self.cells[i] > removed {
                        out.cells[i] -= 1;
                    }
                }
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{chain_rng, standard_normal, uniform_index};

    fn random_tessellation(
        p: usize,
        b: usize,
        d: usize,
        rng: &mut crate::rng::Rng,
    ) -> Tessellation {
        let mut dims = Vec::new();
        while dims.len() < d {
            let c = uniform_index(p, rng);
            if !dims.contains(&c) {
                dims.push(c);
            }
        }
        let centres = (0..b * d).map(|_| 0.5 * standard_normal(rng)).collect();
        let mus = (0..b).map(|_| standard_normal(rng)).collect();
        Tessellation {
            centres,
            dims,
            mus,
            betas: Vec::new(),
            tau: None,
        }
    }

    fn random_data(n: usize, p: usize, rng: &mut crate::rng::Rng) -> Data {
        Data::new((0..n * p).map(|_| standard_normal(rng)).collect(), n, p).unwrap()
    }

    #[test]
    fn nearest_is_lowest_index_on_ties() {
        let g = Geometry::euclidean(1);
        let t = Tessellation {
            centres: vec![0.0, 0.0, 1.0],
            dims: vec![0],
            mus: vec![1.0, 2.0, 3.0],
            betas: Vec::new(),
            tau: None,
        };
        assert_eq!(t.nearest(&[0.0], &g), (0, 0.0));
        assert_eq!(t.nearest(&[0.75], &g), (2, 0.0625));
        assert_eq!(t.value_at(&[0.75], &g), 3.0);
        assert_eq!(t.n_cells(), 3);
        assert_eq!(t.n_dims(), 1);
    }

    #[test]
    fn incremental_updates_equal_full_recompute() {
        use crate::geometry::Metric;
        let mut rng = chain_rng(11);
        for round in 0..400 {
            let p = 1 + uniform_index(4, &mut rng);
            let d = 1 + uniform_index(p, &mut rng);
            let b = 1 + uniform_index(5, &mut rng);
            let x = random_data(2 + uniform_index(30, &mut rng), p, &mut rng);
            // Even rounds Euclidean; odd rounds every column spherical on
            // one sphere, the coordinates already within the radian ranges.
            let g = if round % 2 == 0 {
                Geometry::euclidean(p)
            } else {
                Geometry::structure(&vec![Metric::Spherical { sphere: 0 }; p], p).unwrap()
            };
            let t = random_tessellation(p, b, d, &mut rng);
            let cache = Assignment::full(&x, &t, &g);

            let mut added = t.clone();
            added
                .centres
                .extend((0..d).map(|_| 0.5 * standard_normal(&mut rng)));
            added.mus.push(0.0);
            assert_eq!(
                cache.updated(&x, &added, Delta::CentreAdded, &g),
                Assignment::full(&x, &added, &g)
            );

            let moved_index = uniform_index(b, &mut rng);
            let mut moved = t.clone();
            for v in &mut moved.centres[moved_index * d..(moved_index + 1) * d] {
                *v = 0.5 * standard_normal(&mut rng);
            }
            assert_eq!(
                cache.updated(&x, &moved, Delta::CentreMoved(moved_index), &g),
                Assignment::full(&x, &moved, &g)
            );

            if b >= 2 {
                let removed_index = uniform_index(b, &mut rng);
                let mut removed = t.clone();
                removed
                    .centres
                    .drain(removed_index * d..(removed_index + 1) * d);
                removed.mus.remove(removed_index);
                assert_eq!(
                    cache.updated(&x, &removed, Delta::CentreRemoved(removed_index), &g),
                    Assignment::full(&x, &removed, &g)
                );
            }
        }
    }

    #[test]
    fn soft_incremental_updates_equal_full_recompute() {
        let mut rng = chain_rng(19);
        for _ in 0..200 {
            let p = 1 + uniform_index(4, &mut rng);
            let d = 1 + uniform_index(p, &mut rng);
            let b = 1 + uniform_index(5, &mut rng);
            let x = random_data(2 + uniform_index(30, &mut rng), p, &mut rng);
            let g = Geometry::euclidean(p);
            let mut t = random_tessellation(p, b, d, &mut rng);
            t.tau = Some(0.3);
            let cache = Assignment::full(&x, &t, &g);

            let mut added = t.clone();
            added
                .centres
                .extend((0..d).map(|_| 0.5 * standard_normal(&mut rng)));
            added.mus.push(0.0);
            assert_eq!(
                cache.updated(&x, &added, Delta::CentreAdded, &g),
                Assignment::full(&x, &added, &g)
            );

            let moved_index = uniform_index(b, &mut rng);
            let mut moved = t.clone();
            for v in &mut moved.centres[moved_index * d..(moved_index + 1) * d] {
                *v = 0.5 * standard_normal(&mut rng);
            }
            assert_eq!(
                cache.updated(&x, &moved, Delta::CentreMoved(moved_index), &g),
                Assignment::full(&x, &moved, &g)
            );

            if b >= 2 {
                let removed_index = uniform_index(b, &mut rng);
                let mut removed = t.clone();
                removed
                    .centres
                    .drain(removed_index * d..(removed_index + 1) * d);
                removed.mus.remove(removed_index);
                assert_eq!(
                    cache.updated(&x, &removed, Delta::CentreRemoved(removed_index), &g),
                    Assignment::full(&x, &removed, &g)
                );
            }
        }
    }

    #[test]
    fn soft_weights_reproduce_the_soft_value() {
        let mut rng = chain_rng(23);
        for _ in 0..100 {
            let p = 1 + uniform_index(3, &mut rng);
            let d = 1 + uniform_index(p, &mut rng);
            let b = 1 + uniform_index(5, &mut rng);
            let n = 2 + uniform_index(20, &mut rng);
            let x = random_data(n, p, &mut rng);
            let g = Geometry::euclidean(p);
            let mut t = random_tessellation(p, b, d, &mut rng);
            let tau = 0.1 + 0.5 * standard_normal(&mut rng).abs();
            t.tau = Some(tau);
            let cache = Assignment::full(&x, &t, &g);
            let weights = cache.soft_weights(tau);
            for i in 0..n {
                let row = &weights[i * b..(i + 1) * b];
                let total: f64 = row.iter().sum();
                assert!((total - 1.0).abs() < 1e-12, "{total}");
                // The hard winner carries the largest weight.
                let winner = cache.cells[i];
                assert!(row.iter().all(|&w| w <= row[winner] + 1e-15));
                let value: f64 = row.iter().zip(&t.mus).map(|(&w, &mu)| w * mu).sum();
                let direct = t.value_at(x.row(i), &g);
                assert!((value - direct).abs() < 1e-12, "{value} vs {direct}");
            }
        }
    }

    #[test]
    fn a_tiny_bandwidth_recovers_the_hard_value() {
        let g = Geometry::euclidean(1);
        let mut t = Tessellation {
            centres: vec![0.0, 1.0],
            dims: vec![0],
            mus: vec![2.0, -3.0],
            betas: Vec::new(),
            tau: Some(1e-3),
        };
        assert!((t.value_at(&[0.1], &g) - 2.0).abs() < 1e-9);
        assert!((t.value_at(&[0.9], &g) + 3.0).abs() < 1e-9);
        // A large bandwidth approaches the unweighted mean.
        t.tau = Some(1e3);
        assert!((t.value_at(&[0.1], &g) + 0.5).abs() < 1e-4);
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn serde_keeps_the_bandwidth() {
        let t = Tessellation {
            centres: vec![0.1, 0.2],
            dims: vec![0],
            mus: vec![1.0, -1.0],
            betas: Vec::new(),
            tau: Some(0.2),
        };
        let json = serde_json::to_string(&t).unwrap();
        assert!(json.contains(r#""tau":0.2"#), "{json}");
        assert_eq!(serde_json::from_str::<Tessellation>(&json).unwrap(), t);
        for bad in [
            r#"{"centres":[0.1],"dims":[0],"mus":[1.0],"tau":0.0}"#,
            r#"{"centres":[0.1],"dims":[0],"mus":[1.0],"tau":-0.2}"#,
        ] {
            assert!(serde_json::from_str::<Tessellation>(bad).is_err());
        }
    }

    #[test]
    fn serde_validates() {
        let t = Tessellation {
            centres: vec![0.1, 0.2],
            dims: vec![2],
            mus: vec![1.0, -1.0],
            betas: Vec::new(),
            tau: None,
        };
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(serde_json::from_str::<Tessellation>(&json).unwrap(), t);
        for bad in [
            r#"{"centres":[0.1],"dims":[2],"mus":[1.0,-1.0]}"#,
            r#"{"centres":[0.1,0.2],"dims":[2,2],"mus":[1.0]}"#,
            r#"{"centres":[],"dims":[],"mus":[]}"#,
        ] {
            assert!(serde_json::from_str::<Tessellation>(bad).is_err());
        }
    }
}
