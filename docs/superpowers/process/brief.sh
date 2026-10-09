#!/usr/bin/env bash
# Origin: the SP2 controller's job scratch dir (brief.sh), copied 2026-10-09 into docs/superpowers/process/.
# SP2 template: extracts one task's brief from the plan parts. Adapt WS and PLAN (env vars or the
# defaults below) to the next sub-project's workspace and plan dir; the logic is unchanged from SP2.
#
# Usage: brief.sh TASK_ID   (e.g. T1.1) -> writes $WS/task-<ID>-brief.md
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
WS=${WS:-.superpowers/sdd/2026-10-08-sp2-game-model}
P=${PLAN:-docs/superpowers/plans/2026-10-08-sp2-game-model}
id=$1
part=$(grep -l -E "^### Task ${id//./\\.}:" $P/0[1-4]-*.md | head -1)
[ -n "$part" ] || { echo "task $id not found" >&2; exit 3; }
out=$WS/task-$id-brief.md
{
  echo "# Brief for Task $id (from $part)"
  echo
  echo "Also read: $WS/constraints.md (global constraints, shadow-package strategy, binding contract amendments)."
  echo "The full interface contract is in $P/00-overview.md (section 'Interface contract'); grep it for names from other tasks."
  echo
  echo "## Part preamble"
  awk '/^```/{f=!f} !f && /^### Task /{exit} {print}' "$part"
  echo
  echo "## Task text"
  awk -v id="$id" '
    /^```/ { infence = !infence }
    !infence && /^### Task / { intask = ($0 ~ ("^### Task " id ":")) }
    !infence && /^## / && intask && $0 !~ /^### / { intask = 0 }
    intask { print }
  ' "$part"
  echo
  echo "## Contract notes of this part (accepted unless overridden by the binding amendments in constraints.md)"
  awk '/^## Contract notes/{p=1} p{print}' "$part"
} > "$out"
echo "wrote $out: $(wc -l < "$out") lines"
