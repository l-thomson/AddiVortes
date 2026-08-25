#!/usr/bin/env bash
# Posts a markdown file as one comment on a pull request, editing the
# comment it posted before rather than adding another, so a pull request
# carries one instruction-count table however many times it is pushed.
#
# Usage: tools/perf-instructions-comment.sh <pr-number> <file>
#
# Needs gh authenticated for the repository: GH_TOKEN in CI with
# pull-requests: write.

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <pr-number> <file>" >&2
    exit 2
fi

pr=$1
file=$2
marker='<!-- perf-instructions -->'
repo=${GITHUB_REPOSITORY:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}

body=$(printf '%s\n\n' "$marker"; cat "$file")

existing=$(gh api "repos/$repo/issues/$pr/comments" --paginate \
    --jq ".[] | select(.body | startswith(\"$marker\")) | .id" | head -n 1)

if [ -n "$existing" ]; then
    gh api --method PATCH "repos/$repo/issues/comments/$existing" \
        -f body="$body" --silent
else
    gh api "repos/$repo/issues/$pr/comments" -f body="$body" --silent
fi
