//! The six structural Metropolis-Hastings moves on a tessellation (Stone
//! and Gosling 2025, s. 2.3 and Appendix B) and their selection table.
//!
//! Each move supplies a proposal, the structural change for the assignment
//! cache, and ln[prior ratio x within-move proposal ratio]. The selection
//! ratio ln q(reverse | proposed) - ln q(move | current) is computed from the
//! table, so the boundary corrections of Appendix B (+- ln 2) follow from the
//! weight folding rather than being written by hand.

use crate::maths;
use crate::rng::{standard_normal, uniform, uniform_index, Rng};
use crate::tessellation::{Delta, Tessellation};

/// The move kinds, in selection-table order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Move {
    AddCentre,
    RemoveCentre,
    AddDimension,
    RemoveDimension,
    Change,
    Swap,
}

const MOVES: [Move; 6] = [
    Move::AddCentre,
    Move::RemoveCentre,
    Move::AddDimension,
    Move::RemoveDimension,
    Move::Change,
    Move::Swap,
];

/// Selection weights before folding (Stone and Gosling 2025, Appendix B).
const WEIGHTS: [f64; 6] = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1];

/// Read-only model state the moves consult.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Prior {
    /// Number of covariates p.
    pub p: usize,
    /// Dimension-count prior parameter omega (theta = omega / p).
    pub omega: f64,
    /// Cell-count prior rate lambda_c.
    pub lambda_c: f64,
    /// Centre-coordinate standard deviation sigma_c.
    pub sigma_c: f64,
}

impl Prior {
    /// ln P(b) - ln P(b - 1) of the cell-count prior b - 1 ~ Poisson(lambda_c),
    /// at the larger count b >= 2: ln lambda_c - ln(b - 1).
    fn log_cell_count_ratio(&self, b: usize) -> f64 {
        maths::ln(self.lambda_c) - maths::ln((b - 1) as f64)
    }

    /// ln P(d) - ln P(d - 1) of the dimension-count prior
    /// d - 1 ~ Binomial(p - 1, theta), theta = omega / p, at the larger count
    /// d >= 2: ln(p - d + 1) - ln(d - 1) + ln theta - ln(1 - theta). The
    /// p - 1 trials make the uniform-subset factor 1 / C(p, d) cancel the
    /// uniform covariate pick, so the ratio below is complete.
    fn log_dim_count_ratio(&self, d: usize) -> f64 {
        let p = self.p as f64;
        let d = d as f64;
        let theta = self.omega / p;
        maths::ln(p - d + 1.0) - maths::ln(d - 1.0) + maths::ln(theta) - maths::ln(1.0 - theta)
    }

    /// One centre coordinate, N(0, sigma_c^2), scaled space.
    pub(crate) fn coordinate(&self, rng: &mut Rng) -> f64 {
        self.sigma_c * standard_normal(rng)
    }
}

impl Move {
    /// Whether the move can be proposed from `t`.
    fn valid(self, t: &Tessellation, prior: &Prior) -> bool {
        match self {
            Move::AddCentre | Move::Change => true,
            Move::RemoveCentre => t.n_cells() >= 2,
            Move::AddDimension | Move::Swap => t.n_dims() < prior.p,
            Move::RemoveDimension => t.n_dims() >= 2,
        }
    }

    /// The move whose weight an invalid move folds into (Appendix B).
    fn fold_target(self) -> Option<Move> {
        match self {
            Move::RemoveCentre => Some(Move::AddCentre),
            Move::RemoveDimension => Some(Move::AddDimension),
            Move::AddDimension => Some(Move::RemoveDimension),
            Move::Swap => Some(Move::Change),
            Move::AddCentre | Move::Change => None,
        }
    }

    fn reverse(self) -> Move {
        match self {
            Move::AddCentre => Move::RemoveCentre,
            Move::RemoveCentre => Move::AddCentre,
            Move::AddDimension => Move::RemoveDimension,
            Move::RemoveDimension => Move::AddDimension,
            Move::Change => Move::Change,
            Move::Swap => Move::Swap,
        }
    }

    fn index(self) -> usize {
        MOVES.iter().position(|&m| m == self).expect("listed")
    }
}

/// Selection probabilities in state `t`: invalid moves fold their weight
/// into their partner, then the valid mass is normalised.
pub(crate) fn selection_probs(t: &Tessellation, prior: &Prior) -> [f64; 6] {
    let valid: Vec<bool> = MOVES.iter().map(|m| m.valid(t, prior)).collect();
    let mut probs = [0.0_f64; 6];
    for (i, &m) in MOVES.iter().enumerate() {
        if valid[i] {
            probs[i] += WEIGHTS[i];
        } else if let Some(target) = m.fold_target() {
            let j = target.index();
            if valid[j] {
                probs[j] += WEIGHTS[i];
            }
        }
    }
    let total: f64 = probs.iter().sum();
    for q in &mut probs {
        *q /= total;
    }
    probs
}

/// Draw a move for state `t` with one uniform (ascending cumulative walk).
/// AddCentre and Change are always valid, so a move always exists.
pub(crate) fn select(t: &Tessellation, prior: &Prior, rng: &mut Rng) -> Move {
    let probs = selection_probs(t, prior);
    let target = uniform(rng);
    let mut cumulative = 0.0;
    for (i, &q) in probs.iter().enumerate() {
        cumulative += q;
        if target < cumulative {
            return MOVES[i];
        }
    }
    MOVES[probs
        .iter()
        .rposition(|&q| q > 0.0)
        .expect("some move is valid")]
}

/// A proposed tessellation with the change it makes to the assignment and
/// ln[prior ratio x proposal ratio] excluding the selection ratio. Cell
/// means in the proposal are placeholders; the sampler redraws every mean
/// after accept or reject.
pub(crate) struct Proposal {
    pub tessellation: Tessellation,
    pub delta: Delta,
    pub log_structure_ratio: f64,
}

/// Propose `m` from `t`. Draw order: the move's picks, then coordinates in
/// centre then dimension order.
pub(crate) fn propose(m: Move, t: &Tessellation, prior: &Prior, rng: &mut Rng) -> Proposal {
    let (b, d) = (t.n_cells(), t.n_dims());
    match m {
        // The new centre's coordinates are drawn from their prior, which
        // cancels the proposal density; the reverse uniform pick 1 / (b + 1)
        // cancels against the b + 1 orderings of the enlarged centre set.
        // What remains is the cell-count prior ratio.
        Move::AddCentre => {
            let mut centres = t.centres.clone();
            centres.extend((0..d).map(|_| prior.coordinate(rng)));
            let mut mus = t.mus.clone();
            mus.push(0.0);
            Proposal {
                tessellation: Tessellation {
                    centres,
                    dims: t.dims.clone(),
                    mus,
                },
                delta: Delta::CentreAdded,
                log_structure_ratio: prior.log_cell_count_ratio(b + 1),
            }
        }
        Move::RemoveCentre => {
            let removed = uniform_index(b, rng);
            let mut centres = t.centres.clone();
            centres.drain(removed * d..(removed + 1) * d);
            let mut mus = t.mus.clone();
            mus.remove(removed);
            Proposal {
                tessellation: Tessellation {
                    centres,
                    dims: t.dims.clone(),
                    mus,
                },
                delta: Delta::CentreRemoved(removed),
                log_structure_ratio: -prior.log_cell_count_ratio(b),
            }
        }
        // The incoming covariate is picked uniformly among the p - d unused
        // ones and each centre gets one coordinate from the prior.
        Move::AddDimension => {
            let unused: Vec<usize> = (0..prior.p).filter(|c| !t.dims.contains(c)).collect();
            let incoming = unused[uniform_index(unused.len(), rng)];
            let mut dims = t.dims.clone();
            dims.push(incoming);
            let mut centres = Vec::with_capacity(b * (d + 1));
            for k in 0..b {
                centres.extend_from_slice(t.centre(k));
                centres.push(prior.coordinate(rng));
            }
            Proposal {
                tessellation: Tessellation {
                    centres,
                    dims,
                    mus: t.mus.clone(),
                },
                delta: Delta::Full,
                log_structure_ratio: prior.log_dim_count_ratio(d + 1),
            }
        }
        Move::RemoveDimension => {
            let out = uniform_index(d, rng);
            let mut dims = t.dims.clone();
            dims.remove(out);
            let mut centres = Vec::with_capacity(b * (d - 1));
            for k in 0..b {
                for (j, &c) in t.centre(k).iter().enumerate() {
                    if j != out {
                        centres.push(c);
                    }
                }
            }
            Proposal {
                tessellation: Tessellation {
                    centres,
                    dims,
                    mus: t.mus.clone(),
                },
                delta: Delta::Full,
                log_structure_ratio: -prior.log_dim_count_ratio(d),
            }
        }
        // Counts unchanged; the pick is 1 / b both ways and the old and new
        // coordinate priors cancel the proposal densities.
        Move::Change => {
            let cell = uniform_index(b, rng);
            let mut centres = t.centres.clone();
            for c in &mut centres[cell * d..(cell + 1) * d] {
                *c = prior.coordinate(rng);
            }
            Proposal {
                tessellation: Tessellation {
                    centres,
                    dims: t.dims.clone(),
                    mus: t.mus.clone(),
                },
                delta: Delta::CentreMoved(cell),
                log_structure_ratio: 0.0,
            }
        }
        // Outgoing pick 1 / d, incoming pick 1 / (p - d), mirrored by the
        // reverse; the subset prior is uniform.
        Move::Swap => {
            let out = uniform_index(d, rng);
            let unused: Vec<usize> = (0..prior.p).filter(|c| !t.dims.contains(c)).collect();
            let incoming = unused[uniform_index(unused.len(), rng)];
            let mut dims = t.dims.clone();
            dims[out] = incoming;
            let mut centres = t.centres.clone();
            for k in 0..b {
                centres[k * d + out] = prior.coordinate(rng);
            }
            Proposal {
                tessellation: Tessellation {
                    centres,
                    dims,
                    mus: t.mus.clone(),
                },
                delta: Delta::Full,
                log_structure_ratio: 0.0,
            }
        }
    }
}

/// ln q(reverse | proposed) - ln q(move | current).
pub(crate) fn log_selection_ratio(
    m: Move,
    current: &Tessellation,
    proposed: &Tessellation,
    prior: &Prior,
) -> f64 {
    let forward = selection_probs(current, prior)[m.index()];
    let reverse = selection_probs(proposed, prior)[m.reverse().index()];
    maths::ln(reverse) - maths::ln(forward)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::chain_rng;

    fn prior(p: usize) -> Prior {
        Prior {
            p,
            omega: 3.0_f64.min(p as f64),
            lambda_c: 5.0,
            sigma_c: 0.8,
        }
    }

    fn tess(b: usize, dims: Vec<usize>) -> Tessellation {
        let d = dims.len();
        Tessellation {
            centres: vec![0.0; b * d],
            dims,
            mus: vec![0.0; b],
        }
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    #[test]
    fn folded_table_matches_appendix_b() {
        let pr = prior(10);
        // Interior state: the published weights.
        assert_eq!(
            selection_probs(&tess(3, vec![0, 1]), &pr),
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
        );
        // One cell: RemoveCentre folds into AddCentre.
        let one_cell = selection_probs(&tess(1, vec![0, 1]), &pr);
        close(one_cell[0], 0.4);
        close(one_cell[1], 0.0);
        // One dimension: RemoveDimension folds into AddDimension.
        let one_dim = selection_probs(&tess(3, vec![0]), &pr);
        close(one_dim[2], 0.4);
        close(one_dim[3], 0.0);
        // All dimensions: AddDimension folds into RemoveDimension and Swap
        // into Change.
        let full = selection_probs(&tess(3, (0..10).collect()), &pr);
        close(full[2], 0.0);
        close(full[3], 0.4);
        close(full[4], 0.2);
        close(full[5], 0.0);
    }

    #[test]
    fn selection_ratio_gives_the_boundary_corrections() {
        let pr = prior(10);
        let ln2 = std::f64::consts::LN_2;
        // AddCentre from one cell to two: ln(0.2 / 0.4).
        close(
            log_selection_ratio(
                Move::AddCentre,
                &tess(1, vec![0, 1]),
                &tess(2, vec![0, 1]),
                &pr,
            ),
            -ln2,
        );
        // RemoveCentre from two cells to one: ln(0.4 / 0.2).
        close(
            log_selection_ratio(
                Move::RemoveCentre,
                &tess(2, vec![0, 1]),
                &tess(1, vec![0, 1]),
                &pr,
            ),
            ln2,
        );
        // AddDimension from one dimension to two: -ln 2; to the last
        // dimension: +ln 2.
        close(
            log_selection_ratio(
                Move::AddDimension,
                &tess(2, vec![0]),
                &tess(2, vec![0, 1]),
                &pr,
            ),
            -ln2,
        );
        let all: Vec<usize> = (0..10).collect();
        close(
            log_selection_ratio(
                Move::AddDimension,
                &tess(2, all[..9].to_vec()),
                &tess(2, all.clone()),
                &pr,
            ),
            ln2,
        );
        // Change and Swap in the interior: 0.
        close(
            log_selection_ratio(
                Move::Change,
                &tess(2, vec![0, 1]),
                &tess(2, vec![0, 1]),
                &pr,
            ),
            0.0,
        );
    }

    #[test]
    fn count_prior_ratios_match_the_closed_forms() {
        let pr = prior(10);
        close(pr.log_cell_count_ratio(4), (5.0_f64 / 3.0).ln());
        // d - 1 ~ Binomial(9, 0.3): P(d = 3) / P(d = 2) = (9 - 2 + 1) / 2 * 0.3 / 0.7.
        close(pr.log_dim_count_ratio(3), (8.0 / 2.0 * 0.3 / 0.7_f64).ln());
        // At omega = p the ratio is +infinity: the dimension count saturates.
        let saturated = Prior {
            p: 2,
            omega: 2.0,
            ..pr
        };
        assert_eq!(saturated.log_dim_count_ratio(2), f64::INFINITY);
    }

    #[test]
    fn proposals_have_the_declared_shape() {
        let pr = prior(5);
        let mut rng = chain_rng(3);
        let t = tess(3, vec![1, 4]);
        for _ in 0..50 {
            let m = select(&t, &pr, &mut rng);
            let prop = propose(m, &t, &pr, &mut rng);
            let new = &prop.tessellation;
            assert_eq!(new.centres.len(), new.n_cells() * new.n_dims());
            match m {
                Move::AddCentre => assert_eq!(new.n_cells(), 4),
                Move::RemoveCentre => assert_eq!(new.n_cells(), 2),
                Move::AddDimension => assert_eq!(new.n_dims(), 3),
                Move::RemoveDimension => assert_eq!(new.n_dims(), 1),
                Move::Change => assert_eq!((new.n_cells(), new.n_dims()), (3, 2)),
                Move::Swap => {
                    assert_eq!(new.n_dims(), 2);
                    assert_ne!(new.dims, t.dims);
                }
            }
            let mut dims = new.dims.clone();
            dims.sort_unstable();
            dims.dedup();
            assert_eq!(dims.len(), new.n_dims());
            assert!(dims.iter().all(|&c| c < 5));
        }
    }

    #[test]
    fn add_and_remove_structure_ratios_are_negatives() {
        let pr = prior(6);
        let mut rng = chain_rng(9);
        let t = tess(3, vec![0, 2]);
        let add = propose(Move::AddCentre, &t, &pr, &mut rng);
        let remove = propose(Move::RemoveCentre, &add.tessellation, &pr, &mut rng);
        close(add.log_structure_ratio, -remove.log_structure_ratio);
        let add = propose(Move::AddDimension, &t, &pr, &mut rng);
        let remove = propose(Move::RemoveDimension, &add.tessellation, &pr, &mut rng);
        close(add.log_structure_ratio, -remove.log_structure_ratio);
    }
}
