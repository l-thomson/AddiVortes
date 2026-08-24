#!/usr/bin/env bash
# Same-machine A/B of the wall-clock benchmarks over two revisions.
#
# Usage: tools/perf-compare.sh <rev-a> <rev-b> [filter]
#
# Both revisions are built and run in one session on one machine into a
# shared target directory, then compared with critcmp. Wall-clock numbers
# taken on different machines or in different sessions are not comparable,
# so no stored history exists and no gate reads these numbers.
#
# Revision A runs twice, first and last. The A-against-A table is the
# drift check: a machine that is not quiet shows a difference there, and a
# comparison taken on it means nothing. Discard the run and close whatever
# was competing for the core.
#
# Requires critcmp (cargo install critcmp).

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 <rev-a> <rev-b> [filter]" >&2
    exit 2
fi

rev_a=$1
rev_b=$2
filter=${3:-}

command -v critcmp >/dev/null || {
    echo "critcmp not found: cargo install critcmp" >&2
    exit 1
}

root=$(git rev-parse --show-toplevel)
work=$root/target/perf
target=$work/target
mkdir -p "$work"

sha_a=$(git rev-parse --verify "$rev_a^{commit}")
sha_b=$(git rev-parse --verify "$rev_b^{commit}")

checkouts=()
cleanup() {
    for dir in "${checkouts[@]}"; do
        git worktree remove --force "$dir" 2>/dev/null || true
    done
}
trap cleanup EXIT

checkout() {
    local sha=$1 dir=$work/src-$2
    rm -rf "$dir"
    git worktree add --detach --quiet "$dir" "$sha"
    checkouts+=("$dir")
    echo "$dir"
}

dir_a=$(checkout "$sha_a" a)
dir_b=$(checkout "$sha_b" b)

# One baseline name per run, so the two revisions and the repeat sit side
# by side in one criterion directory for critcmp to read.
run() {
    local dir=$1 baseline=$2
    echo "== $baseline ==" >&2
    (
        cd "$dir"
        CARGO_TARGET_DIR=$target cargo bench --locked \
            --manifest-path bench/Cargo.toml \
            --bench wall_clock -- --save-baseline "$baseline" $filter
    )
}

run "$dir_a" a
run "$dir_b" b
run "$dir_a" a-repeat

echo
echo "$rev_a ($sha_a) against $rev_b ($sha_b)"
critcmp --target-dir "$target" a b

echo
echo "drift: $rev_a against itself, first run against last"
critcmp --target-dir "$target" a a-repeat
