# The experimental items in plain English

A companion to [experimental.md](experimental.md), which is the status
table. This page says the same things without the vocabulary: what each
gated option does, where the idea came from, and what evidence exists
that it works. The audience is a reader who has not read the paper.
Where the two pages disagree the table is authoritative.

## What "experimental" means here

The crate ships with these thirteen options switched off. They are in
the source, but a normal build cannot reach them: the Cargo feature
`experimental` has to be turned on deliberately, and the R and Python
packages never turn it on.

Two things are promised about them, and one is withheld.

Promised: they are tested to the same standard as the published method,
and enabling the feature does not change the answers of any
configuration that uses none of them. A reader who ignores these
options gets bit-for-bit the same numbers either way.

Withheld: the usual stability guarantee. A later release may rename an
option or change the numbers it produces, with a changelog line. That
is why they are separated out rather than simply shipped: the line
between what is settled and what is still moving is drawn in the build
system, not in a footnote.

## The checks referred to below

Each check catches a class of mistake the others cannot. The first five
are named once here and referenced by name in each entry; the sixth
applies to the crate as a whole rather than to individual items. The
technical statement of each is in [testing.md](testing.md).

**Reduction test.** Every option has a setting at which it should
behave identically to something already trusted: a distance measure
with its dial set to the ordinary one, a censored-data model given data
with nothing censored. The test runs both and requires the same numbers
to come out, draw for draw, rather than merely similar answers. A
companion test requires the option to change the answers at its other
settings, so a feature cannot pass by being accidentally inert.

**Known answer.** For deliberately simplified cases the right answer
can be obtained a different way, by numerical integration rather than
by simulation. The sampler has to match that independent calculation.
This catches a program that is self-consistent but aimed at the wrong
target.

**Rank check (simulation-based calibration).** Invent a truth at
random, manufacture data from it, then ask the sampler to recover the
truth it was never told, and record where the real value sits among the
draws. Repeated hundreds of times, an honest sampler puts the true
value in every position equally often. Bias in either direction shows
up as a lopsided pile-up. Talts et al. (2018); Modrak et al. (2025).

**Two-route check (Geweke).** There are two routes to generating
data-and-truth pairs from one set of assumptions: one direct, one
through the sampler's machinery. If the machinery is right the two
routes are statistically indistinguishable. Comparing them catches
errors the rank check can miss. Geweke (2004).

**Sabotage check.** A test that passes says nothing unless it would
have failed. Three deliberately mispriced samplers live in the source,
one of them a reproduction of an arithmetic error the upstream R
package corrected in 0.6.8, and the checks above are required to reject
every one of them. This is how the checks earn the right to be
believed.

**Upstream match.** Results are compared against the original authors'
R package, CRAN AddiVortes 0.6.9, on standard datasets, within the
margin two runs of any random method would differ by. It shows this
implementation has not drifted from the published one; it cannot catch
a mistake the two share. It does not apply item by item, because the
upstream package has none of these options: it holds the published
method steady underneath them. The AFT entry below has the closest
analogue, an informational run beside the BART package.

Options are validated at one of two grades. An option that only
changes a definition is validated as a component: reduction test,
round-trip of the configuration, and a small rank check. An option that
estimates something, and so changes the answers, is validated as a
model: known answer where one exists, rank check and two-route check at
both the fast size and the nightly size, and the sabotage check where a
mispriced version can be written.

## Group one: new ways of measuring "nearby"

The method carves the data into cells around a set of centre points,
and each observation belongs to the centre it is nearest to. "Nearest"
therefore does a great deal of work. The published method measures it
as ordinary straight-line distance, with special handling for spheres
and for categories. These five options offer other definitions, all
long established elsewhere in statistics. None changes the model, only
the ruler.

### Minkowski distance, and Manhattan

A dial that changes how distance is added up across several
measurements. At one end it is straight-line distance, as before; at
the other it is city-block distance, measured the way a walk on a
street grid is measured, adding each leg separately instead of cutting
the corner. Turning the dial down makes a cell less dominated by one
measurement being far out.

Based on the Minkowski family of distances, textbook material in
cluster analysis. Manhattan is the name for the setting p = 1, offered
separately because that is what people call it.

Checked by the reduction test in two directions: at p = 2 the dial is
straight-line distance and the runs must be identical, and Manhattan
must agree exactly with p = 1. A small rank check at the Manhattan
setting, and tests that saving and reloading a fitted model preserves
the choice.

### Cosine distance

Compares direction rather than position: two rows count as close when
their pattern of values points the same way, whatever the magnitudes.
Useful when the shape of a profile matters and its size does not.

Based on cosine similarity, the standard measure in text and
recommender work.

Checked by the reduction and round-trip tests and a small rank check.
Two limitations are documented rather than tested away: this is not a
true distance, because going by way of a third point can be shorter,
and the crate's rescaling of each column makes the origin it measures
direction from depend on the training data. The option is intended for
covariates that are directions already.

### Gower distance

A way of measuring closeness when the data mix numbers with categories,
such as age and income beside job title and region. Each column is
scored in its own natural way and the scores are averaged, so no column
dominates through having larger units.

Based on Gower (1971), the standard answer to mixed-type data. Gower's
optional weighting for missing values is not implemented.

Checked by the reduction and round-trip tests, a small rank check, and
a test that an invalid category code is refused at the start of the fit
with a named error rather than quietly producing a number.

### Mahalanobis distance

Straight-line distance that allows for measurements moving together. If
height and weight rise in step, someone tall and heavy is unremarkable
while someone tall and light is genuinely unusual; ordinary distance
cannot tell those apart and this can. The pattern of co-movement is
supplied by the user, not learnt.

Based on Mahalanobis distance, a foundational measure in multivariate
statistics.

Checked by the reduction test: supply the pattern that says nothing
moves together and the result must be bit-identical to ordinary
distance. A small rank check, and tests that a missing, mis-shaped or
mathematically impossible pattern is refused at the start of the fit.

### Mixing measures across columns

Lets different parts of one dataset use different rulers, city-block
for one group of columns and direction-based for another, combined into
a single notion of nearby.

No new theory: the published method already adds distances across
columns, and this exposes that arithmetic as a choice.

Checked by keeping each member measure's own reduction test, plus the
identity that a group of one column reduces exactly to the plain case.
A small rank check on the combination and round-trip tests.

## Group two: which covariates a cell may use

Each cell looks at only a handful of the available covariates. The
published method picks that handful with no preferences at all. These
two options change how the choice is made, which matters when there are
many covariates and few of them carry signal.

### Weighted inclusion

The user says in advance which covariates are expected to matter, as a
set of weights. A higher weight means the sampler reaches for that
covariate more often; a weight of zero shuts it out entirely. It is a
way of using knowledge that falls short of a hard rule.

Based on the equivalent feature in bartMachine (Kapelner and Bleich
2016), a widely used package in the tree-based family.

Checked by the reduction test: equal weights must give the
no-preferences default back exactly, and the code takes the same path
in that case, so the two cannot drift apart. A zero weight is tested to
exclude its covariate, and a small rank check follows. The lighter
grade is deliberate: nothing here is estimated. The weights are fixed
numbers the user supplies, so there is no new estimation that could be
wrong.

### DART inclusion

The same idea with the weights learnt instead of supplied. The sampler
starts even-handed and concentrates on the covariates earning their
place, ignoring the rest. This is the standard remedy for data with
many covariates of which few are real.

Based on Linero (2018), implemented to match the settings the BART R
package ships under `sparse = TRUE`, so results are comparable with
existing practice.

Validated at model grade, because the weights are now estimated and so
change the answers: rank check and two-route check at both sizes, and
the sabotage check. In the configuration surface this is one component
among many; the promotion to model grade is a judgement recorded in the
source, on the ground that anything sampled deserves it.

## Group three: the shape of the answer inside a cell

In the published method a cell holds one flat value and an observation
belongs wholly to one cell, so the fitted surface is a set of plateaus
with hard steps between them. Both options here soften that, and both
change the model, so both are validated at model grade.

### Linear cell basis

Instead of one flat value per cell, the value is allowed to slope
across the cell. Where the truth is a smooth trend, a few sloping cells
do the work of many flat ones, which is a coarser and more
interpretable description of the same pattern.

Based on the standard step from constant to linear pieces, familiar
from linear-model leaves in the regression-tree literature. The
arithmetic follows the rules the published method already uses.

Checked by known answer, rank check and two-route check at both sizes,
and the sabotage check. A guard rail goes with it: a slope is only
meaningful when the columns share a scale, so the fit is refused when
they do not rather than returning a plausible wrong answer.

### Soft membership

Removes the hard borders. An observation near a boundary belongs partly
to both neighbouring cells instead of jumping between them, with a
width controlling how blurred the border is, and the width is learnt
from the data. The result is a smooth surface in place of a staircase,
which usually predicts better when the truth is smooth.

Based on SBART, Linero and Yang (2018), which did this for tree splits;
the same idea is carried to the assignment of rows to cells. The
default width matches the SoftBart package's default, so the behaviour
is familiar.

Checked by known answer, rank check and two-route check at both sizes,
and the sabotage check. The reduction test carries the most weight
here: the hard-edged default must produce a bit-identical run to the
plain sampler. Two limits are documented and enforced rather than
approximated: soft membership cannot be combined with the linear cell
basis, or with the varying-spread model, because neither combination
has a derived update, so the crate refuses those configurations.

## Group four: new kinds of response

The published method predicts a plain number, or a yes and no. These
four handle responses that come up constantly in real work and are
handled badly by treating them as plain numbers. Each is a textbook
model; what is new is fitting it with tessellations rather than trees.

### Tobit outcome

For measurements that cannot pass a known boundary and so stack up on
it: spending that cannot fall below zero, an instrument that cannot
read above its maximum. Treating those piled-up values as ordinary
readings biases everything; this model reads them as "at least this
much" or "at most this much".

Based on the type-I tobit model, fitted by the data augmentation of
Chib (1992).

Checked by the reduction test, where data that never reaches the limit
must reproduce the Gaussian model draw for draw; known answer against
numerical integration of the censored likelihood; rank check and
two-route check at both sizes.

### AFT outcome

For time-to-event data, where the defining problem is that the study
ends before everyone has had the event. A patient is known to have been
alive at five years without a date of death being known. Dropping those
rows discards the best cases, and counting five years as the survival
time is simply wrong; this model uses them as the partial information
they are.

Based on the lognormal accelerated failure time model (Wei 1992),
matching the `abart` function of the BART R package, deliberately, so
the two are comparable.

Checked by the reduction test against the Gaussian model on log times
when nothing is censored; known answer against numerical integration;
rank check and two-route check at both sizes. It is additionally run
beside `abart` on the same data. That comparison is reported for
information and not asserted as a test: the priors of the two packages
differ, so the answers should be close without being equal, and
asserting equality would be a test that fails for the wrong reason.

### Interval-censored outcome

For responses recorded as a range rather than a value: income given as
a band, a condition known to have developed between two appointments, a
reading below a detection threshold. The usual fudge is to take the
midpoint, which invents precision that is not there. This model uses
the range, and handles rows that are exact, two-sided or open-ended
mixed together.

Based on the standard interval-censored likelihood, fitted by the same
augmentation as the tobit model with a two-sided truncated draw.

Checked by the reduction test, where ranges of zero width, meaning
exact values, must reproduce the Gaussian model draw for draw; known
answer against numerical integration of the interval likelihood; rank
check and two-route check at both sizes.

### Ordinal outcome

For responses with an order but no arithmetic: mild, moderate, severe,
or a rating from one to five. Scoring these one to five and averaging
assumes the step from one to two is the step from four to five, which
is rarely true. This model keeps the order and estimates the spacing
between the levels from the data instead of assuming it.

Based on the ordered probit model of Albert and Chib (1993, s. 5), with
the level boundaries updated by the blocked move of Cowles (1996), a
technique introduced because the one-at-a-time alternative explores
badly.

The most heavily checked of the four. The reduction test requires that
two categories reproduce the probit model draw for draw. Known answer
against numerical integration for both the boundaries and the cell
values; rank check and two-route check at both sizes with the
boundaries among the test quantities; the sabotage check; and, at full
size, a check that the boundaries are being explored at a reasonable
rate rather than crawling, which is the specific failure the Cowles
move exists to prevent, verified rather than assumed.

## What none of this proves

[testing.md](testing.md) is blunt about the limits and they apply here
unchanged. Each layer catches a class of defect the others cannot and
none proves correctness on its own. A pass at these sizes is evidence,
not proof. The upstream comparison cannot detect a defect shared with
the original implementation. The simulation recovery tolerances are
loose by design. The rank and two-route checks run at a fast size on
every change and a larger size nightly; larger still would be better.

An option stops being experimental by the stabilisation rule stated
once in the crate-root documentation, argued in a pull request against
that rule rather than by flipping a switch. Each row of
[experimental.md](experimental.md) links the pull request that added the
item, which is its public record.

## References

The works these options come from, in the order they appear above.

- Gower, J. C. (1971). A general coefficient of similarity and some of
  its properties. Biometrics 27(4), 857-871.
- Kapelner, A. and Bleich, J. (2016). bartMachine: machine learning
  with Bayesian additive regression trees. Journal of Statistical
  Software 70(4).
- Linero, A. R. (2018). Bayesian regression trees for high-dimensional
  prediction and variable selection. Journal of the American
  Statistical Association 113(522), 626-636.
- Linero, A. R. and Yang, Y. (2018). Bayesian regression tree ensembles
  that adapt to smoothness and sparsity. Journal of the Royal
  Statistical Society Series B 80(5), 1087-1110.
- Chib, S. (1992). Bayes inference in the Tobit censored regression
  model. Journal of Econometrics 51(1-2), 79-99.
- Wei, L. J. (1992). The accelerated failure time model: a useful
  alternative to the Cox regression model in survival analysis.
  Statistics in Medicine 11(14-15), 1871-1879.
- Albert, J. H. and Chib, S. (1993). Bayesian analysis of binary and
  polychotomous response data. Journal of the American Statistical
  Association 88(422), 669-679.
- Cowles, M. K. (1996). Accelerating Monte Carlo Markov chain
  convergence for cumulative-link generalized linear models.
  Statistics and Computing 6, 101-111.

The works behind the checks are listed in [testing.md](testing.md).
