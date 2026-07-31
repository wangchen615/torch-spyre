#!/usr/bin/env bash
# Create the KV-cache offload backlog on torch-spyre/torch-spyre:
#   - kvc-offloading label
#   - 3 epics + 18 child issues (bodies in ./bodies/)
#   - milestone 2026 Q3 on all
#   - cross-links (Blocked by #N) substituted into bodies
#   - native sub-issue nesting (children under epics)
#   - optional project-board placement (needs `project` scope; skipped if absent)
#
# Requires: gh with WRITE (push) on the repo, and (for the board) `project` scope.
# Re-runnable: skips creating an issue whose exact title already exists (matched open+closed).
#
# Usage:
#   cd scripts/kv_offload_issues && bash create_issues.sh
set -euo pipefail

REPO="torch-spyre/torch-spyre"
OWNER="torch-spyre"
PROJECT_NUMBER=2
MILESTONE="2026 Q3"
LABEL_KVC="kvc-offloading"
LABEL_EPIC="epic"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BODIES="$HERE/bodies"

# --- board Status option names (edit if your board uses different labels) ---
STATUS_EPIC="Epic"
STATUS_BLOCKED="Blocked"
STATUS_READY="Ready"
# (Review / Done are set by PR state later, not here.)

# key -> body file (basename without .md) and human title
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

# key -> body filename (basename)
declare -A BODYFILE=(
  [EDESIGN]="EPIC-DESIGN" [ED1]="E-D1" [ED2]="E-D2" [ED3]="E-D3"
  [M1]="EPIC-M1" [M1P1]="M1-P1" [M1P2]="M1-P2" [M1P3]="M1-P3"
  [M1F1]="M1-F1" [M1F2]="M1-F2" [M1T1]="M1-T1" [M1T2]="M1-T2" [M1T3]="M1-T3"
  [M1S1]="M1-S1" [M1S2]="M1-S2" [M1S3]="M1-S3"
  [M2]="EPIC-M2" [M2F1]="M2-F1" [M2F2]="M2-F2" [M2F3]="M2-F3"
  [M2T1]="M2-T1" [M2T2]="M2-T2" [M2S1]="M2-S1" [M2S2]="M2-S2" [M2S3]="M2-S3"
)

# creation order: children before epics so epics can (optionally) reference them,
# and dependencies before dependents (not strictly required — we patch #s afterwards).
ORDER=(ED1 ED2 ED3 \
       M1P1 M1P2 M1P3 M1F1 M1F2 M1T1 M1T2 M1T3 M1S1 M1S2 M1S3 \
       M2F1 M2F2 M2F3 M2T1 M2T2 M2S1 M2S2 M2S3 \
       EDESIGN M1 M2)

EPICS=(EDESIGN M1 M2)
declare -A CHILDREN=(
  [EDESIGN]="ED1 ED2 ED3"
  [M1]="M1P1 M1P2 M1P3 M1F1 M1F2 M1T1 M1T2 M1T3 M1S1 M1S2 M1S3"
  [M2]="M2F1 M2F2 M2F3 M2T1 M2S1 M2S2 M2S3"
)

declare -A NUM   # key -> issue number

echo "==> Step 1: ensure label '$LABEL_KVC' exists"
if ! gh label list --repo "$REPO" --search "$LABEL_KVC" | grep -q "$LABEL_KVC"; then
  gh label create "$LABEL_KVC" --repo "$REPO" \
    --description "KV-cache offload (host-RAM / shared host memory pool) work" \
    --color 1d76db
else
  echo "    label already exists"
fi

find_existing () {  # $1 = exact title -> echoes number or nothing
  gh issue list --repo "$REPO" --state all --limit 200 --json number,title \
    --jq ".[] | select(.title == \$t) | .number" --arg t "$1" 2>/dev/null | head -1
}

echo "==> Step 2: create issues (labels + milestone), body has placeholders for now"
for key in "${ORDER[@]}"; do
  title="${TITLE[$key]}"
  body="$(cat "$BODIES/${BODYFILE[$key]}.md")"
  existing="$(find_existing "$title")"
  if [[ -n "$existing" ]]; then
    echo "    [$key] exists as #$existing (skip create)"
    NUM[$key]="$existing"
    continue
  fi
  labels="$LABEL_KVC"
  for e in "${EPICS[@]}"; do [[ "$key" == "$e" ]] && labels="$labels,$LABEL_EPIC"; done
  # M2T2 is deferred backlog: created (so it is tracked + cross-linked from the M2 epic) but
  # assigned NO milestone. Everything else gets the milestone.
  if [[ "$key" == "M2T2" ]]; then
    url="$(gh issue create --repo "$REPO" --title "$title" --body "$body" --label "$labels")"
  else
    url="$(gh issue create --repo "$REPO" --title "$title" --body "$body" \
            --label "$labels" --milestone "$MILESTONE")"
  fi
  n="${url##*/}"
  NUM[$key]="$n"
  echo "    [$key] created #$n"
done

echo "==> Step 3: build placeholder->#N substitution and patch every body"
SED_ARGS=()
for key in "${!NUM[@]}"; do
  SED_ARGS+=(-e "s/__${key}__/#${NUM[$key]}/g")
done
for key in "${ORDER[@]}"; do
  patched="$(sed "${SED_ARGS[@]}" "$BODIES/${BODYFILE[$key]}.md")"
  gh issue edit "${NUM[$key]}" --repo "$REPO" --body "$patched" >/dev/null
  echo "    [$key] #${NUM[$key]} body cross-links patched"
done

echo "==> Step 4: nest children under epics as native sub-issues"
issue_gql_id () { gh api "repos/$REPO/issues/$1" --jq '.node_id'; }
for epic in "${EPICS[@]}"; do
  parent_id="$(issue_gql_id "${NUM[$epic]}")"
  for child in ${CHILDREN[$epic]}; do
    child_id="$(issue_gql_id "${NUM[$child]}")"
    if gh api graphql -f query='
      mutation($p:ID!, $c:ID!) {
        addSubIssue(input:{issueId:$p, subIssueId:$c}) { issue { number } }
      }' -f p="$parent_id" -f c="$child_id" >/dev/null 2>&1; then
      echo "    #${NUM[$child]} nested under epic #${NUM[$epic]}"
    else
      echo "    !! sub-issue link failed for #${NUM[$child]} -> #${NUM[$epic]} (feature/perm?); task-list checklist still links it"
    fi
  done
done

echo "==> Step 5 (optional): add to project board + set Status column"
PROJ_ID="$(gh api graphql -f query='
  query($o:String!,$n:Int!){ organization(login:$o){ projectV2(number:$n){ id
    field(name:"Status"){ ... on ProjectV2SingleSelectField { id options { id name } } } } } }' \
  -f o="$OWNER" -F n="$PROJECT_NUMBER" --jq '.data.organization.projectV2.id' 2>/dev/null || true)"

if [[ -z "$PROJ_ID" || "$PROJ_ID" == "null" ]]; then
  echo "    project not accessible (need 'project' scope). SKIPPING board."
  echo "    -> run: gh auth refresh -s project --hostname github.com  then re-run this script."
else
  STATUS_FIELD_ID="$(gh api graphql -f query='
    query($o:String!,$n:Int!){ organization(login:$o){ projectV2(number:$n){
      field(name:"Status"){ ... on ProjectV2SingleSelectField { id } } } } }' \
    -f o="$OWNER" -F n="$PROJECT_NUMBER" --jq '.data.organization.projectV2.field.id')"
  opt_id () {  # $1 = option name
    gh api graphql -f query='
      query($o:String!,$n:Int!){ organization(login:$o){ projectV2(number:$n){
        field(name:"Status"){ ... on ProjectV2SingleSelectField { options { id name } } } } } }' \
      -f o="$OWNER" -F n="$PROJECT_NUMBER" \
      --jq ".data.organization.projectV2.field.options[] | select(.name==\$x) | .id" --arg x "$1"
  }
  OPT_EPIC="$(opt_id "$STATUS_EPIC")"; OPT_BLOCKED="$(opt_id "$STATUS_BLOCKED")"; OPT_READY="$(opt_id "$STATUS_READY")"

  # which keys are blocked = have any non-"none" Blocked-by (i.e. body mentions "Blocked by:" + a placeholder key)
  is_blocked () {
    case "$1" in
      ED1|ED2|ED3|M1P1|M1T1) return 1 ;;   # no blockers (Ready)
      *) return 0 ;;                        # everything else has a blocker (Blocked)
    esac
  }
  add_to_board () {  # $1 = issue number ; echoes project item id
    local iid; iid="$(issue_gql_id "$1")"
    gh api graphql -f query='
      mutation($p:ID!,$c:ID!){ addProjectV2ItemById(input:{projectId:$p, contentId:$c}){ item { id } } }' \
      -f p="$PROJ_ID" -f c="$iid" --jq '.data.addProjectV2ItemById.item.id'
  }
  set_status () {  # $1 = item id ; $2 = option id
    gh api graphql -f query='
      mutation($p:ID!,$i:ID!,$f:ID!,$o:String!){ updateProjectV2ItemFieldValue(
        input:{projectId:$p, itemId:$i, fieldId:$f, value:{singleSelectOptionId:$o}}){ projectV2Item { id } } }' \
      -f p="$PROJ_ID" -f i="$1" -f f="$STATUS_FIELD_ID" -f o="$2" >/dev/null
  }
  for key in "${ORDER[@]}"; do
    item="$(add_to_board "${NUM[$key]}")"
    if printf '%s\n' "${EPICS[@]}" | grep -qx "$key"; then
      set_status "$item" "$OPT_EPIC";    echo "    #${NUM[$key]} -> Epic"
    elif is_blocked "$key"; then
      set_status "$item" "$OPT_BLOCKED"; echo "    #${NUM[$key]} -> Blocked"
    else
      set_status "$item" "$OPT_READY";   echo "    #${NUM[$key]} -> Ready"
    fi
  done
  echo "    NOTE: Review/Done columns are driven by PR state; move an issue there when its PR opens/merges."
fi

echo "==> DONE. Numbers:"
for key in "${ORDER[@]}"; do printf '    %-8s #%s\n' "$key" "${NUM[$key]}"; done
