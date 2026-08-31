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

FLEX_SRC="$HOME/dt-inductor/flex/flex/include"
FLEX_LIB="$HOME/dt-inductor/build/senbfcc/flex"
SENTIENT="$HOME/dt-inductor/sentient"

export LD_LIBRARY_PATH="$FLEX_LIB:$LD_LIBRARY_PATH"
export CMAKE_INCLUDE_PATH="$FLEX_SRC:$SENTIENT/deeptools/include:$SENTIENT/runtime/include:$SENTIENT/spyre_comms/include:/opt/ibm/spyre/senlib/include"
export CMAKE_LIBRARY_PATH="$FLEX_LIB:$SENTIENT/deeptools/lib:$SENTIENT/runtime/lib"
export USE_SPYRE_CCL="${USE_SPYRE_CCL:-0}"
export SPYRE_COMMS_INSTALL_DIR="${SPYRE_COMMS_INSTALL_DIR:-$SENTIENT/spyre_comms}"
