# Contributing

## Setup

Rust stable; the minimum supported version is 1.74. Build and test with

    cargo test --locked

## Gates

Every pull request must pass, and CI enforces:

    cargo fmt --all --check
    cargo clippy --all-targets --locked -- -D warnings
    cargo test --locked
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --locked
    cargo deny check advisories licenses bans sources

Each gate also runs with `--features experimental`. The full-size
statistical suite runs nightly; run it locally with

    cargo nextest run --locked --features experimental --run-ignored all

The full-size calibration tests write ranks and samples under
`target/calibration` (override with `CALIBRATION_DIR`);
`benchmarks/calibration/evaluate.R` turns them into rank ECDF difference
and comparison plots, as the nightly `calibration` job does.

## Reproducibility and snapshots

The reproducibility contract is in the crate-root documentation
([crates/thiessen/src/lib.rs](crates/thiessen/src/lib.rs)). Fixed-seed
chains under `crates/thiessen/tests/snapshots/` are bit-exact on
`x86_64-unknown-linux-gnu`; other targets check posterior summaries
within Monte Carlo error.

Regenerate snapshots with `cargo insta review` (or
`INSTA_UPDATE=always cargo test`) on `x86_64-unknown-linux-gnu` only. A
pull request that regenerates a snapshot carries a minor version bump and
a changelog line "Sampled values changed" with the reason.

## Stable and experimental

The published method is stable: the models and components of Stone and
Gosling (2025) and of CRAN AddiVortes. Everything else is experimental
and is compiled only with the `experimental` Cargo feature, under
`#[cfg(feature = "experimental")]`, with a row in
[docs/experimental.md](docs/experimental.md). Experimental items meet the
same test bar (known answer where one exists, calibration at two sizes,
snapshot, documentation) and are outside the semver promise.

Commits adding or changing experimental items use the scope
`feat(experimental):` or `fix(experimental):`; their changelog lines start
with "(experimental)". A sampled-value change to an experimental item
takes the "Sampled values changed" line but does not force a minor bump.
Release notes carry the sentence "Options behind the `experimental`
feature are outside the semver promise; see docs/experimental.md".

An item graduates when it has met the full bar, has a citable write-up
with a DOI stating the model, priors, calibration and recovery evidence,
has shipped behind the feature for one minor release, and its tracking
issue has no open questions. The graduation pull request removes the
`cfg` gate, moves the table row to graduated, and is a minor version
bump.

## Pull requests

Branch from `dev`; pull requests squash-merge into `dev` with a green
`alls-green` status. Commit messages follow Conventional Commits. The
template's four boxes (tests, docs, changelog, breaking or
sampled-values change) are the whole checklist.

## Releases

Component tags: `core-vX.Y.Z` for the crate, `py-vX.Y.Z` for the Python
package, `r-vX.Y.Z` for the R package. Each binding versions
independently and states the core version it wraps (pyproject metadata
for Python; `Config/thiessen/core-version` in DESCRIPTION for R).

Core releases run through release-plz (`release-plz.toml`, the
release-plz workflow): a release PR carries the version bump and the
changelog section; merging it creates the tag, publishes to crates.io
and opens a GitHub release with the changelog section as notes. A
Python tag triggers the wheel matrix and trusted publishing. R releases
follow the CRAN checklist, with r-universe as the development channel.

Versions follow semver. Patch releases preserve sampled values for a
fixed seed; minor releases may change them and the changelog entry says
"Sampled values changed" with the reason, following the value-stability
policy of rand. The same rule continues past 1.0: a sampled-value
change is a minor bump with the same line, never silent.

Every GitHub release is archived on Zenodo with a DOI once the Zenodo
integration is enabled; the concept DOI badge joins the README at the
first release. `CITATION.cff` carries the software and paper references
and is validated in CI; the R package's `inst/CITATION` and the DOI in
the Python documentation join with their packages.

At each tag, `main` is fast-forwarded to the tagged commit on `dev`
with `git push origin <tag>^{commit}:main`; `main` carries releases
only. `CHANGELOG.md` is keepachangelog; the R package keeps `NEWS.md`;
Python release notes come from the changelog. The first tag of every
component is gated on the final name; the working name is never
published.

## Triage

Issues are triaged by the maintainer; bug reports need the version,
platform, seed and minimal configuration the template asks for.
Questions belong in Discussions.
