#!/usr/bin/env bash
# TRIAGE-ONLY pass: create the 21 KV-offload issues with just what triage permits.
#   - full bodies (from ./bodies/)
#   - existing `epic` label on the 3 epics (no kvc-offloading label — needs write)
#   - NO milestone (needs write)
#   - cross-links: real #N substituted into "Blocked by" lines + epic checklists
#   - epic bodies keep their `- [ ]` child checklists (task-list tracking, like #778)
#   - NO native sub-issue nesting, NO project board (both need write / project scope)
#
# A write-capable teammate can later run create_issues.sh (the full version) — it is
# idempotent (skips issues whose exact title already exists) and will add the
# kvc-offloading label, 2026 Q3 milestone, native sub-issues, and board placement.
#
# Usage: cd scripts/kv_offload_issues && bash create_issues_triage.sh
set -euo pipefail

REPO="torch-spyre/torch-spyre"
LABEL_EPIC="epic"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BODIES="$HERE/bodies"

declare -A TITLE=(
  [EDESIGN]="[Epic] KV-cache offload design docs (runtime + torch-spyre + spyre-inference)"
  [ED1]="[E-D1] Merge the hardware-runtime shared host memory KV pool RFC"
  [ED2]="[E-D2] Merge the torch-spyre Python-surface design doc"
  [ED3]="[E-D3] Merge the spyre-inference upstream-connector-port RFC"
  [M1]="[Epic] Milestone 1 — host-RAM KV offload end-to-end (shared pool)"
  [M1P1]="[M1-P1] Prereq: reproducible custom-built hardware runtime + torch-spyre + spyre-inference env"
  [M1P2]="[M1-P2] Prereq: spyre-inference recomputation baseline on latest code + pin a version"
  [M1P3]="[M1-P3] Prereq: define the host CPU buffer/tensor model for raw copy"
  [M1F1]="[M1-F1] Hardware runtime: public raw (byte-exact) host<->device DMA"
  [M1F2]="[M1-F2] Hardware runtime: SharedHostPool (shared host memory pool, cross-process)"
  [M1T1]="[M1-T1] torch-spyre: get_composite_address accessor"
  [M1T2]="[M1-T2] torch-spyre: SharedHostPool pybind passthrough (incl. cross-process sharing)"
  [M1T3]="[M1-T3] torch-spyre: copy_tensor_raw(dev_tensor, pool, slot_id, ...) binding (canonical)"
  [M1S1]="[M1-S1] spyre-inference: SpyreKvDmaCopier + kv_offload package scaffold"
  [M1S2]="[M1-S2] spyre-inference: SpyreCpuOffloadingHandlers"
  [M1S3]="[M1-S3] spyre-inference: SpyreOffloadingSpec + registration + M1 acceptance & benchmark"
  [M2]="[Epic] Milestone 2 — cross-instance shared host-memory KV pool"
  [M2F1]="[M2-F1] Hardware runtime: SharedHostMetadata (block-hash -> slot directory)"
  [M2F2]="[M2-F2] Hardware runtime: concurrency protocol — full data-race coverage"
  [M2F3]="[M2-F3] Hardware runtime: copyRaw multi-chunk + cross-process slot round-trip"
  [M2T1]="[M2-T1] torch-spyre: SharedHostMetadata (+ shared-pool attach) pybind passthroughs"
  [M2T2]="[M2-T2] torch-spyre: get_dma_stream accessor (DEFERRED backlog — no milestone)"
  [M2S1]="[M2-S1] spyre-inference: SpyreSharedOffloadingSpec + registration"
  [M2S2]="[M2-S2] spyre-inference: shared-pool round-trip + torn-read test"
  [M2S3]="[M2-S3] spyre-inference: cross-instance test + M2 acceptance & benchmark"
)
declare -A BODYFILE=(
  [EDESIGN]="EPIC-DESIGN" [ED1]="E-D1" [ED2]="E-D2" [ED3]="E-D3"
  [M1]="EPIC-M1" [M1P1]="M1-P1" [M1P2]="M1-P2" [M1P3]="M1-P3"
  [M1F1]="M1-F1" [M1F2]="M1-F2" [M1T1]="M1-T1" [M1T2]="M1-T2" [M1T3]="M1-T3"
  [M1S1]="M1-S1" [M1S2]="M1-S2" [M1S3]="M1-S3"
  [M2]="EPIC-M2" [M2F1]="M2-F1" [M2F2]="M2-F2" [M2F3]="M2-F3"
  [M2T1]="M2-T1" [M2T2]="M2-T2" [M2S1]="M2-S1" [M2S2]="M2-S2" [M2S3]="M2-S3"
)
ORDER=(ED1 ED2 ED3 \
       M1P1 M1P2 M1P3 M1F1 M1F2 M1T1 M1T2 M1T3 M1S1 M1S2 M1S3 \
       M2F1 M2F2 M2F3 M2T1 M2T2 M2S1 M2S2 M2S3 \
       EDESIGN M1 M2)
EPICS=(EDESIGN M1 M2)
declare -A NUM

find_existing () {
  gh issue list --repo "$REPO" --state all --limit 300 --json number,title \
    --jq ".[] | select(.title == \$t) | .number" --arg t "$1" 2>/dev/null | head -1
}
is_epic () { printf '%s\n' "${EPICS[@]}" | grep -qx "$1"; }

echo "==> Creating 21 issues (triage mode: epic label only, no milestone/nesting/board)"
for key in "${ORDER[@]}"; do
  title="${TITLE[$key]}"
  body="$(cat "$BODIES/${BODYFILE[$key]}.md")"
  existing="$(find_existing "$title")"
  if [[ -n "$existing" ]]; then
    echo "    [$key] exists as #$existing (skip)"; NUM[$key]="$existing"; continue
  fi
  if is_epic "$key"; then
    url="$(gh issue create --repo "$REPO" --title "$title" --body "$body" --label "$LABEL_EPIC")"
  else
    url="$(gh issue create --repo "$REPO" --title "$title" --body "$body")"
  fi
  n="${url##*/}"; NUM[$key]="$n"
  echo "    [$key] created #$n${url:+  ($url)}"
done

echo "==> Patching cross-links (#N) into every body"
SED_ARGS=()
for key in "${!NUM[@]}"; do SED_ARGS+=(-e "s/__${key}__/#${NUM[$key]}/g"); done
for key in "${ORDER[@]}"; do
  patched="$(sed "${SED_ARGS[@]}" "$BODIES/${BODYFILE[$key]}.md")"
  gh issue edit "${NUM[$key]}" --repo "$REPO" --body "$patched" >/dev/null
  echo "    [$key] #${NUM[$key]} patched"
done

echo "==> DONE (triage). Numbers:"
for key in "${ORDER[@]}"; do printf '    %-8s #%s\n' "$key" "${NUM[$key]}"; done
echo
echo "STILL TODO (needs a write-capable account):"
echo "  - create 'kvc-offloading' label + apply to all 21"
echo "  - set milestone '2026 Q3' on all 21"
echo "  - native sub-issue nesting (children under epics)"
echo "  - project board (view 23): Epic / Blocked / Ready columns"
echo "  -> run create_issues.sh with a write + project token; it is idempotent and will"
echo "     skip re-creating these issues and just add the missing pieces."
