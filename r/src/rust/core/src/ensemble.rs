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

/// The standard deviation of the soft-membership bandwidth proposal on
/// ln tau.
const BANDWIDTH_STEP: f64 = 0.5;

/// The tessellations of one cell family and their running total at the
/// training rows.
#[derive(Debug, Clone)]
pub(crate) struct Ensemble<F: CellFamily> {
    family: F,
    prior: Prior,
    /// Rate of the exponential prior on the soft-membership bandwidth;
    /// `None` under hard membership.
    bandwidth_rate: Option<f64>,
    tessellations: Vec<Tessellation>,
    assignments: Vec<Assignment>,
    /// The combined value of the ensemble at each training row.
    total: Vec<f64>,
}

impl<F: CellFamily> Ensemble<F> {
    /// `m` single-cell tessellations on one covariate each, every cell
    /// holding `cell_value`, the total set to `total` at every row. Draw
    /// order per tessellation: the covariate, then the centre coordinate,
    /// then, under soft membership, the bandwidth from its prior.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        family: F,
        prior: Prior,
        bandwidth_rate: Option<f64>,
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
                let tau = bandwidth_rate.map(|rate| -maths::ln(1.0 - rng::uniform(rng)) / rate);
                Tessellation {
                    centres: vec![centre],
                    dims: vec![dim],
                    mus: vec![cell_value],
                    betas: Vec::new(),
                    tau,
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
            bandwidth_rate,
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

    /// The cell of every training row under each tessellation.
    pub(crate) fn assignments(&self) -> &[Assignment] {
        &self.assignments
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

    fn partials(
        &self,
        j: usize,
        x: &Data,
        input: &[f64],
        weights: &[f64],
        soft: Option<&[f64]>,
    ) -> Vec<Partial> {
        let current = &self.tessellations[j];
        let cells = &self.assignments[j].cells;
        let b = current.n_cells();
        (0..input.len())
            .map(|i| {
                let own = match soft {
                    Some(w) => soft_value(&current.mus, &w[i * b..(i + 1) * b]),
                    None => current.value_in_cell(cells[i], x.row(i)),
                };
                self.family
                    .partial(input[i], weights[i], self.total[i], own)
            })
            .collect()
    }

    /// The backfitting update of tessellation `j`: the partials against the
    /// rest of the ensemble, one structural move with the empty-cell guard,
    /// under soft membership one bandwidth move, the cell values, and the
    /// running total.
    pub(crate) fn backfit(
        &mut self,
        j: usize,
        x: &Data,
        input: &[f64],
        weights: &[f64],
        rng: &mut Rng,
        #[cfg(test)] breakage: crate::broken::Breakage,
    ) {
        let tau = self.tessellations[j].tau;
        let mut soft_weights = tau.map(|tau| self.assignments[j].soft_weights(tau));
        let partials = self.partials(j, x, input, weights, soft_weights.as_deref());
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
        let proposed_weights = tau.map(|tau| proposed_assignment.soft_weights(tau));
        let proposed_stats = self.family.accumulate(
            x,
            &proposal.tessellation,
            &proposed_assignment.cells,
            &partials,
            proposal.tessellation.n_cells(),
            proposed_weights.as_deref(),
        );
        // A proposal leaving a cell empty is rejected before the acceptance
        // draw, so no uniform is consumed.
        let mut stats = None;
        if proposed_stats.all_occupied() {
            let current_stats = self.family.accumulate(
                x,
                current,
                cells,
                &partials,
                current.n_cells(),
                soft_weights.as_deref(),
            );
            #[allow(unused_mut)]
            let mut log_alpha = self.family.log_marginal(
                &proposed_stats,
                #[cfg(test)]
                breakage,
            ) - self.family.log_marginal(
                &current_stats,
                #[cfg(test)]
                breakage,
            ) + proposal.log_structure_ratio
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
                soft_weights = proposed_weights;
                stats = Some(proposed_stats);
            } else {
                stats = Some(current_stats);
            }
        }
        if tau.is_some() {
            self.update_bandwidth(
                j,
                x,
                &partials,
                &mut stats,
                &mut soft_weights,
                rng,
                #[cfg(test)]
                breakage,
            );
        }
        self.redraw(j, x, &partials, stats, input, soft_weights.as_deref(), rng);
    }

    /// The bandwidth move of soft tessellation `j`: a random-walk
    /// Metropolis step on ln tau with the cell values integrated out,
    /// prior tau ~ Exponential(rate). Draw order: the proposal normal,
    /// then the acceptance uniform.
    #[allow(clippy::too_many_arguments)]
    fn update_bandwidth(
        &mut self,
        j: usize,
        x: &Data,
        partials: &[Partial],
        stats: &mut Option<F::Stats>,
        soft_weights: &mut Option<Vec<f64>>,
        rng: &mut Rng,
        #[cfg(test)] breakage: crate::broken::Breakage,
    ) {
        let rate = self
            .bandwidth_rate
            .expect("a soft ensemble carries a bandwidth prior");
        let tau = self.tessellations[j].tau.expect("a soft tessellation");
        let current_stats = stats.take().unwrap_or_else(|| {
            self.family.accumulate(
                x,
                &self.tessellations[j],
                &self.assignments[j].cells,
                partials,
                self.tessellations[j].n_cells(),
                soft_weights.as_deref(),
            )
        });
        let proposed_tau = tau * maths::exp(BANDWIDTH_STEP * rng::standard_normal(rng));
        let proposed_weights = self.assignments[j].soft_weights(proposed_tau);
        let proposed_stats = self.family.accumulate(
            x,
            &self.tessellations[j],
            &self.assignments[j].cells,
            partials,
            self.tessellations[j].n_cells(),
            Some(&proposed_weights),
        );
        // The exponential prior ratio and the Jacobian of the log-scale
        // walk, tau' / tau.
        let log_alpha = self.family.log_marginal(
            &proposed_stats,
            #[cfg(test)]
            breakage,
        ) - self.family.log_marginal(
            &current_stats,
            #[cfg(test)]
            breakage,
        ) - rate * (proposed_tau - tau)
            + maths::ln(proposed_tau)
            - maths::ln(tau);
        debug_assert!(!log_alpha.is_nan());
        if maths::ln(rng::uniform(rng)) < log_alpha {
            self.tessellations[j].tau = Some(proposed_tau);
            *soft_weights = Some(proposed_weights);
            *stats = Some(proposed_stats);
        } else {
            *stats = Some(current_stats);
        }
    }

    /// The cell values of tessellation `j` given its current structure,
    /// then the running total.
    #[allow(clippy::too_many_arguments)]
    fn redraw(
        &mut self,
        j: usize,
        x: &Data,
        partials: &[Partial],
        stats: Option<F::Stats>,
        input: &[f64],
        soft: Option<&[f64]>,
        rng: &mut Rng,
    ) {
        let tessellation = &mut self.tessellations[j];
        let cells = &self.assignments[j].cells;
        let stats = stats.unwrap_or_else(|| {
            self.family.accumulate(
                x,
                tessellation,
                cells,
                partials,
                tessellation.n_cells(),
                soft,
            )
        });
        let (values, slopes) = self.family.draw(&stats, rng);
        tessellation.mus = values;
        tessellation.betas = slopes;
        let b = tessellation.n_cells();
        for i in 0..input.len() {
            let own = match soft {
                Some(w) => soft_value(&tessellation.mus, &w[i * b..(i + 1) * b]),
                None => tessellation.value_in_cell(cells[i], x.row(i)),
            };
            self.total[i] = self.family.total(input[i], &partials[i], own);
        }
    }
}

/// The kernel-weighted sum of the cell values at one observation.
fn soft_value(mus: &[f64], weights: &[f64]) -> f64 {
    mus.iter().zip(weights).map(|(&mu, &w)| mu * w).sum()
}

#[cfg(test)]
impl<F: CellFamily> Ensemble<F> {
    /// One sweep with the structural moves disabled: every tessellation's
    /// cell values given its current structure. On fixed tessellations the
    /// chain is the conjugate Gibbs sampler of the known-answer tests.
    pub(crate) fn conjugate_sweep(
        &mut self,
        x: &Data,
        input: &[f64],
        weights: &[f64],
        rng: &mut Rng,
    ) {
        for j in 0..self.tessellations.len() {
            let tau = self.tessellations[j].tau;
            let soft = tau.map(|tau| self.assignments[j].soft_weights(tau));
            let partials = self.partials(j, x, input, weights, soft.as_deref());
            self.redraw(j, x, &partials, None, input, soft.as_deref(), rng);
        }
    }

    /// Replace tessellation `j` and rebuild its cache; the caller resets
    /// the total.
    pub(crate) fn set_tessellation(&mut self, j: usize, x: &Data, t: Tessellation, total: f64) {
        self.assignments[j] = Assignment::full(x, &t, &self.prior.geometry);
        self.tessellations[j] = t;
        self.total.iter_mut().for_each(|v| *v = total);
    }
}
