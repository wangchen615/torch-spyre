#!/usr/bin/env bash
# Build environment for the kvc-offload-dev branch against the locally built
# flex kvc-offload-dev stack.
#
# Two shadowing hazards this works around:
#   1. sentient/runtime/lib/libflex.so is stale and lacks copyRaw/SharedHostPool,
#      so the fresh build must come first on LD_LIBRARY_PATH.
#   2. sentient/runtime/include has stale flex headers with no
#      shared_host_pool.hpp and no copyRaw, so the flex source tree must come
#      first on CMAKE_INCLUDE_PATH.
#
# Multi-Spyre (USE_SPYRE_CCL=1) works as of 2026-08-31. It previously failed two
# ways, both now resolved by rebuilding spyre-comms against this flex:
#   - the installed libspyre_comms.so.1 (Aug 19) wanted the 8-arg
#     flex::createDmaParams; current flex exports a 9-arg version.
#   - spyre-comms needed senlib::v2::PinnedMemoryWrapper via a transitive
#     include that flex's header cleanup removed (fixed upstream in
#     spyre-comms #345).
# To rebuild spyre-comms against this flex, see scripts/kvc-build-spyre-comms.sh.
#
# Set USE_SPYRE_CCL=1 in your environment to build with Multi-Spyre; the default
# below leaves it off so single-device builds stay fast.

# PATHS MOVED 2026-08-31 (flex rebase + rebuild):
#   library: build/senbfcc/flex/libflex.so  ->  build/senbfcc/libflex.so
#            (no /flex suffix any more; the old directory is GONE, so a stale
#             path fails loudly rather than silently falling through)
#   headers: flex/flex/include             ->  flex/include
#            (flex/flex/include/flex now holds only the generated flex_export.h,
#             which is a decoy: pointing there yields confusing "no such file"
#             errors instead of a clean failure)
#
# FLEX_SRC below is a STAGED header tree, not the flex working tree. The flex
# checkout is currently on pr1569/yue-round2, which carries only #1569 and so
# has neither shared_host_pool.hpp nor the slot-addressed copyRaw declaration --
# even though the built library exports both. The staging dir is extracted from
# origin/kvc-offload-dev (git archive, no checkout) so the flex working tree is
# left untouched. Refresh it with scripts/kvc-stage-flex-headers.sh.
FLEX_SRC="$HOME/dt-inductor/build/flex-hdr-kvc/include"
FLEX_LIB="$HOME/dt-inductor/build/senbfcc"
SENTIENT="$HOME/dt-inductor/sentient"

# Fail loudly if the expected flex artifacts are missing, rather than letting the
# build fall through to the stale Aug 19 library/headers under sentient/.
if [[ ! -f "$FLEX_LIB/libflex.so" ]]; then
  echo "kvc-build-env: ERROR: no libflex.so at $FLEX_LIB" >&2
  echo "  (did the flex build move again? check build/senbfcc/)" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -f "$FLEX_SRC/flex/memory_interface/shared_host_pool.hpp" ]]; then
  echo "kvc-build-env: ERROR: staged flex headers missing shared_host_pool.hpp" >&2
  echo "  run scripts/kvc-stage-flex-headers.sh" >&2
  return 1 2>/dev/null || exit 1
fi

export LD_LIBRARY_PATH="$FLEX_LIB:$LD_LIBRARY_PATH"
export CMAKE_INCLUDE_PATH="$FLEX_SRC:$SENTIENT/deeptools/include:$SENTIENT/runtime/include:$SENTIENT/spyre_comms/include:/opt/ibm/spyre/senlib/include"
export CMAKE_LIBRARY_PATH="$FLEX_LIB:$SENTIENT/deeptools/lib:$SENTIENT/runtime/lib"
export USE_SPYRE_CCL="${USE_SPYRE_CCL:-0}"
export SPYRE_COMMS_INSTALL_DIR="${SPYRE_COMMS_INSTALL_DIR:-$SENTIENT/spyre_comms}"
