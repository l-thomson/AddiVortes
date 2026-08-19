//! A Voronoi tessellation of a covariate subspace and the assignment of
//! observations to its cells.

use crate::data::Data;
use crate::error::{Error, Result};

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
    /// Cell means mu_1..mu_b (scaled space).
    pub(crate) mus: Vec<f64>,
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

    /// Squared Euclidean distance from `row` (a full p-length scaled row) to
    /// centre `k` over the active covariates.
    pub(crate) fn key(&self, row: &[f64], k: usize) -> f64 {
        let centre = self.centre(k);
        let mut key = 0.0;
        for (&dim, &c) in self.dims.iter().zip(centre) {
            let diff = row[dim] - c;
            key += diff * diff;
        }
        key
    }

    /// The nearest centre to `row` and its key; ties go to the lowest index.
    pub(crate) fn nearest(&self, row: &[f64]) -> (usize, f64) {
        let mut best = f64::INFINITY;
        let mut best_cell = 0;
        for k in 0..self.n_cells() {
            let key = self.key(row, k);
            if key < best {
                best = key;
                best_cell = k;
            }
        }
        (best_cell, best)
    }

    /// The value of the tessellation at `row`: the mean of the cell `row`
    /// falls in.
    pub(crate) fn value_at(&self, row: &[f64]) -> f64 {
        self.mus[self.nearest(row).0]
    }
}

#[derive(serde::Deserialize)]
struct TessellationParts {
    centres: Vec<f64>,
    dims: Vec<usize>,
    mus: Vec<f64>,
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
        if parts
            .centres
            .iter()
            .chain(&parts.mus)
            .any(|v| !v.is_finite())
        {
            return Err(bad("tessellation values must be finite"));
        }
        for (i, dim) in parts.dims.iter().enumerate() {
            if parts.dims[..i].contains(dim) {
                return Err(bad("tessellation dimensions must be distinct"));
            }
        }
        Ok(Self {
            centres: parts.centres,
            dims: parts.dims,
            mus: parts.mus,
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
/// key, for one tessellation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Assignment {
    /// Cell index per observation.
    pub(crate) cells: Vec<usize>,
    /// Squared distance to the assigned centre per observation.
    keys: Vec<f64>,
}

impl Assignment {
    /// Assignment of every row of `x` under `t`, computed in full.
    pub(crate) fn full(x: &Data, t: &Tessellation) -> Self {
        let n = x.n_rows();
        let mut cells = Vec::with_capacity(n);
        let mut keys = Vec::with_capacity(n);
        for i in 0..n {
            let (cell, key) = t.nearest(x.row(i));
            cells.push(cell);
            keys.push(key);
        }
        Self { cells, keys }
    }

    /// Assignment under `new`, which differs from the tessellation this
    /// cache was built for by `delta`. Equal to [`Assignment::full`] on the
    /// same inputs. With dims unchanged an untouched centre's key against
    /// any observation is unchanged, so only pairs involving the touched
    /// centre are recomputed.
    pub(crate) fn updated(&self, x: &Data, new: &Tessellation, delta: Delta) -> Self {
        let n = x.n_rows();
        match delta {
            Delta::Full => Self::full(x, new),
            Delta::CentreAdded => {
                let added = new.n_cells() - 1;
                let mut out = self.clone();
                for i in 0..n {
                    let key = new.key(x.row(i), added);
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
                    if self.cells[i] == moved {
                        let (cell, key) = new.nearest(row);
                        out.cells[i] = cell;
                        out.keys[i] = key;
                    } else {
                        let key = new.key(row, moved);
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
                for i in 0..n {
                    if self.cells[i] == removed {
                        let (cell, key) = new.nearest(x.row(i));
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
        Tessellation { centres, dims, mus }
    }

    fn random_data(n: usize, p: usize, rng: &mut crate::rng::Rng) -> Data {
        Data::new((0..n * p).map(|_| standard_normal(rng)).collect(), n, p).unwrap()
    }

    #[test]
    fn nearest_is_lowest_index_on_ties() {
        let t = Tessellation {
            centres: vec![0.0, 0.0, 1.0],
            dims: vec![0],
            mus: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(t.nearest(&[0.0]), (0, 0.0));
        assert_eq!(t.nearest(&[0.75]), (2, 0.0625));
        assert_eq!(t.value_at(&[0.75]), 3.0);
        assert_eq!(t.n_cells(), 3);
        assert_eq!(t.n_dims(), 1);
    }

    #[test]
    fn incremental_updates_equal_full_recompute() {
        let mut rng = chain_rng(11);
        for _ in 0..200 {
            let p = 1 + uniform_index(4, &mut rng);
            let d = 1 + uniform_index(p, &mut rng);
            let b = 1 + uniform_index(5, &mut rng);
            let x = random_data(2 + uniform_index(30, &mut rng), p, &mut rng);
            let t = random_tessellation(p, b, d, &mut rng);
            let cache = Assignment::full(&x, &t);

            let mut added = t.clone();
            added
                .centres
                .extend((0..d).map(|_| 0.5 * standard_normal(&mut rng)));
            added.mus.push(0.0);
            assert_eq!(
                cache.updated(&x, &added, Delta::CentreAdded),
                Assignment::full(&x, &added)
            );

            let moved_index = uniform_index(b, &mut rng);
            let mut moved = t.clone();
            for v in &mut moved.centres[moved_index * d..(moved_index + 1) * d] {
                *v = 0.5 * standard_normal(&mut rng);
            }
            assert_eq!(
                cache.updated(&x, &moved, Delta::CentreMoved(moved_index)),
                Assignment::full(&x, &moved)
            );

            if b >= 2 {
                let removed_index = uniform_index(b, &mut rng);
                let mut removed = t.clone();
                removed
                    .centres
                    .drain(removed_index * d..(removed_index + 1) * d);
                removed.mus.remove(removed_index);
                assert_eq!(
                    cache.updated(&x, &removed, Delta::CentreRemoved(removed_index)),
                    Assignment::full(&x, &removed)
                );
            }
        }
    }

    #[test]
    fn serde_validates() {
        let t = Tessellation {
            centres: vec![0.1, 0.2],
            dims: vec![2],
            mus: vec![1.0, -1.0],
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
