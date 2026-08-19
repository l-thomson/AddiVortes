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

The full-size statistical suite runs nightly; run it locally with

    cargo nextest run --locked --run-ignored all

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

## Pull requests

Branch from `dev`; pull requests squash-merge into `dev` with a green
`alls-green` status. Commit messages follow Conventional Commits. The
template's four boxes (tests, docs, changelog, breaking or
sampled-values change) are the whole checklist.

## Releases

Component tags `core-vX.Y.Z`, `py-vX.Y.Z`, `r-vX.Y.Z`; the process is
in the release section of this file once the first release is wired.
`main` carries releases only.

## Triage

Issues are triaged by the maintainer; bug reports need the version,
platform, seed and minimal configuration the template asks for.
Questions belong in Discussions.
