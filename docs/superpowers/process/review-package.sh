#!/usr/bin/env bash
# Review package: commit list, stat and full diff (with function context) of BASE..HEAD.
# Usage: review-package.sh BASE HEAD OUTFILE
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
base=$1; head=$2; out=$3
{
  echo "# Review package $base..$head"
  echo
  echo "## Commits"
  git log --format='%h %s%n%b' "$base..$head"
  echo
  echo "## Stat"
  git diff --stat=160 "$base..$head"
  echo
  echo "## Diff"
  echo '```diff'
  git diff -U8 --function-context "$base..$head"
  echo '```'
} > "$out"
echo "wrote $out: $(wc -l < "$out") lines"
