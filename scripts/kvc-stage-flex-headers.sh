#!/usr/bin/env bash
# Stage the flex headers that match the built libflex.so, WITHOUT touching the
# flex working tree.
#
# Why this exists: the flex checkout is often parked on a single-PR branch (e.g.
# pr1569/yue-round2, which carries #1569 only). Such a branch has neither
# shared_host_pool.hpp (#1570's file) nor the slot-addressed copyRaw
# declaration, even when the built library exports both -- because the library
# was built from the integration branch. Compiling torch-spyre against the
# working tree in that state fails with "no such file" or a missing overload,
# which looks like a torch-spyre bug but is a branch mismatch.
#
# Rather than switch someone else's working tree (it may have uncommitted work,
# or be in use by another session), extract the include tree from the
# integration branch with `git archive`, which touches no checked-out file.
set -euo pipefail

FLEX_REPO="${FLEX_REPO:-$HOME/dt-inductor/flex}"
FLEX_REF="${FLEX_REF:-origin/kvc-offload-dev}"
STAGE="${STAGE:-$HOME/dt-inductor/build/flex-hdr-kvc}"

echo "== staging $FLEX_REF include tree -> $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cd "$FLEX_REPO"
git archive "$FLEX_REF" include | tar -x -C "$STAGE"

# flex_export.h is generated at build time, so it is not in the git tree. Take it
# from wherever the build left it in the working tree.
if [[ ! -f "$STAGE/include/flex/flex_export.h" ]]; then
  for cand in "$FLEX_REPO/include/flex/flex_export.h" \
              "$FLEX_REPO/flex/include/flex/flex_export.h"; do
    if [[ -f "$cand" ]]; then
      cp "$cand" "$STAGE/include/flex/"
      echo "   copied generated flex_export.h from ${cand#$HOME/}"
      break
    fi
  done
fi

echo "== verifying the staged tree matches what torch-spyre needs"
fail=0
[[ -f "$STAGE/include/flex/memory_interface/shared_host_pool.hpp" ]] \
  || { echo "   MISSING shared_host_pool.hpp"; fail=1; }
grep -q "SharedHostPool& pool" "$STAGE/include/flex/runtime_stream/runtime_stream.hpp" \
  || { echo "   MISSING slot-addressed copyRaw declaration"; fail=1; }
[[ -f "$STAGE/include/flex/flex_export.h" ]] \
  || { echo "   MISSING generated flex_export.h"; fail=1; }
if (( fail )); then
  echo "== FAILED: $FLEX_REF does not carry the KV-offload surface" >&2
  exit 1
fi

echo "   OK: shared_host_pool.hpp, slot copyRaw, flex_export.h all present"
echo
echo "Now build with:  source scripts/kvc-build-env.sh && python3 setup.py build_ext --inplace -j 16"
