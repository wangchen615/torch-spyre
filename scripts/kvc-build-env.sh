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
# USE_SPYRE_CCL=0 avoids a PRE-EXISTING breakage unrelated to KV offload:
# spyre_comms::initialize_library expects shared_ptr<flex::RuntimeContext> while
# torch-spyre passes a raw pointer. Clean upstream main fails the same way.

FLEX_SRC="$HOME/dt-inductor/flex/flex/include"
FLEX_LIB="$HOME/dt-inductor/build/senbfcc/flex"
SENTIENT="$HOME/dt-inductor/sentient"

export LD_LIBRARY_PATH="$FLEX_LIB:$LD_LIBRARY_PATH"
export CMAKE_INCLUDE_PATH="$FLEX_SRC:$SENTIENT/deeptools/include:$SENTIENT/runtime/include:$SENTIENT/spyre_comms/include:/opt/ibm/spyre/senlib/include"
export CMAKE_LIBRARY_PATH="$FLEX_LIB:$SENTIENT/deeptools/lib:$SENTIENT/runtime/lib"
export USE_SPYRE_CCL=0
