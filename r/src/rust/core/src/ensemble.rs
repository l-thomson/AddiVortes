//! An ensemble of tessellations over one conjugate cell family: the
//! tessellations, their assignment caches, the running total, and the
//! backfitting sweep of Stone and Gosling (2025, Algorithm 1) with the
//! Metropolis-Hastings structural moves. The Gaussian mean function is an
//! additive ensemble of [`GaussianCells`](crate::cells::GaussianCells); the
//! heteroscedastic variance function is a multiplicative ensemble of
//! [`InverseGammaCells`](crate::cells::InverseGammaCells).

use crate::cells::{CellFamily, Partial, Stats};
use crate::data::Data;
use crate::geometry::Geometry;
use crate::maths;
use crate::moves::{self, Prior};
use crate::rng::{self, Rng};
use crate::tessellation::{Assignment, Tessellation};

/// The tessellations of one cell family and their running total at the
/// training rows.
#[derive(Debug, Clone)]
pub(crate) struct Ensemble<F: CellFamily> {
    family: F,
    prior: Prior,
    tessellations: Vec<Tessellation>,
    assignments: Vec<Assignment>,
    /// The combined value of the ensemble at each training row.
    total: Vec<f64>,
}

impl<F: CellFamily> Ensemble<F> {
    /// `m` single-cell tessellations on one covariate each, every cell
    /// holding `cell_value`, the total set to `total` at every row. Draw
    /// order per tessellation: the covariate, then the centre coordinate.
    pub(crate) fn new(
        family: F,
        prior: Prior,
        x: &Data,
        m: usize,
        cell_value: f64,
        total: f64,
        rng: &mut Rng,
    ) -> Self {
        let tessellations: Vec<Tessellation> = (0..m)
            .map(|_| {
                let dim = prior.initial_dim(rng);
                let centre = prior.coordinate(dim, rng);
                Tessellation {
                    centres: vec![centre],
                    dims: vec![dim],
                    mus: vec![cell_value],
                }
            })
            .collect();
        let assignments = tessellations
            .iter()
            .map(|t| Assignment::full(x, t, &prior.geometry))
            .collect();
        Self {
            family,
            prior,
            tessellations,
            assignments,
            total: vec![total; x.n_rows()],
        }
    }

    #[cfg(test)]
    pub(crate) fn family(&self) -> &F {
        &self.family
    }

    pub(crate) fn tessellations(&self) -> &[Tessellation] {
        &self.tessellations
    }

    pub(crate) fn geometry(&self) -> &Geometry {
        &self.prior.geometry
    }

    pub(crate) fn total(&self) -> &[f64] {
        &self.total
    }

    /// One sweep: every tessellation in turn through [`backfit`](Self::backfit).
    /// Replace the inclusion weights of the structural prior; the DART
    /// update writes the sampled vector here each sweep.
    #[cfg(feature = "experimental")]
    pub(crate) fn set_inclusion_weights(&mut self, weights: crate::moves::InclusionWeights) {
        self.prior.weights = Some(weights);
    }

    pub(crate) fn sweep(
        &mut self,
        x: &Data,
        input: &[f64],
        weights: &[f64],
        rng: &mut Rng,
        #[cfg(test)] breakage: crate::broken::Breakage,
    ) {
        for j in 0..self.tessellations.len() {
            self.backfit(
                j,
                x,
                input,
                weights,
                rng,
                #[cfg(test)]
                breakage,
            );
        }
    }

    fn partials(&self, j: usize, input: &[f64], weights: &[f64]) -> Vec<Partial> {
        let current = &self.tessellations[j];
        let cells = &self.assignments[j].cells;
        (0..input.len())
            .map(|i| {
                self.family
                    .partial(input[i], weights[i], self.total[i], current.mus[cells[i]])
            })
            .collect()
    }

    /// The backfitting update of tessellation `j`: the partials against the
    /// rest of the ensemble, one structural move with the empty-cell guard,
    /// the cell values, and the running total.
    pub(crate) fn backfit(
        &mut self,
        j: usize,
        x: &Data,
        input: &[f64],
        weights: &[f64],
        rng: &mut Rng,
        #[cfg(test)] breakage: crate::broken::Breakage,
    ) {
        let partials = self.partials(j, input, weights);
        let current = &self.tessellations[j];
        let cells = &self.assignments[j].cells;

        let m = moves::select(current, &self.prior, rng);
        let proposal = moves::propose(m, current, &self.prior, rng);
        let proposed_assignment = self.assignments[j].updated(
            x,
            &proposal.tessellation,
            proposal.delta,
            &self.prior.geometry,
        );
        let proposed_stats = self.family.accumulate(
            &proposed_assignment.cells,
            &partials,
            proposal.tessellation.n_cells(),
        );
        // A proposal leaving a cell empty is rejected before the acceptance
        // draw, so no uniform is consumed.
        let mut stats = None;
        if proposed_stats.all_occupied() {
            let current_stats = self.family.accumulate(cells, &partials, current.n_cells());
            #[allow(unused_mut)]
            let mut log_alpha = self.family.log_marginal(&proposed_stats)
                - self.family.log_marginal(&current_stats)
                + proposal.log_structure_ratio
                + moves::log_selection_ratio(m, current, &proposal.tessellation, &self.prior);
            #[cfg(test)]
            {
                log_alpha += crate::broken::log_alpha_shift(
                    breakage,
                    m,
                    current,
                    &proposal.tessellation,
                    &self.prior,
                    self.family.cell_normaliser(),
                );
            }
            debug_assert!(!log_alpha.is_nan());
            let u = rng::uniform(rng);
            if maths::ln(u) < log_alpha {
                self.tessellations[j] = proposal.tessellation;
                self.assignments[j] = proposed_assignment;
                stats = Some(proposed_stats);
            } else {
                stats = Some(current_stats);
            }
        }
        self.redraw(j, &partials, stats, input, rng);
    }

    /// The cell values of tessellation `j` given its current structure,
    /// then the running total.
    fn redraw(
        &mut self,
        j: usize,
        partials: &[Partial],
        stats: Option<F::Stats>,
        input: &[f64],
        rng: &mut Rng,
    ) {
        let tessellation = &mut self.tessellations[j];
        let cells = &self.assignments[j].cells;
        let stats = stats.unwrap_or_else(|| {
            self.family
                .accumulate(cells, partials, tessellation.n_cells())
        });
        tessellation.mus = self.family.draw(&stats, rng);
        for i in 0..input.len() {
            self.total[i] = self
                .family
                .total(input[i], &partials[i], tessellation.mus[cells[i]]);
        }
    }
}

#[cfg(test)]
impl<F: CellFamily> Ensemble<F> {
    /// One sweep with the structural moves disabled: every tessellation's
    /// cell values given its current structure. On fixed tessellations the
    /// chain is the conjugate Gibbs sampler of the known-answer tests.
    pub(crate) fn conjugate_sweep(&mut self, input: &[f64], weights: &[f64], rng: &mut Rng) {
        for j in 0..self.tessellations.len() {
            let partials = self.partials(j, input, weights);
            self.redraw(j, &partials, None, input, rng);
        }
    }

    /// Replace tessellation `j` and rebuild its cache; the caller resets
    /// the total.
    pub(crate) fn set_tessellation(&mut self, j: usize, x: &Data, t: Tessellation, total: f64) {
        self.assignments[j] = Assignment::full(x, &t, &self.prior.geometry);
        self.tessellations[j] = t;
        self.total.iter_mut().for_each(|v| *v = total);
    }

    pub(crate) fn assignments(&self) -> &[Assignment] {
        &self.assignments
    }
}
