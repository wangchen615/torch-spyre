#!/usr/bin/env bash
# Rebuild spyre-comms against the locally built flex kvc-offload-dev stack, so
# Multi-Spyre (USE_SPYRE_CCL=1) works in torch-spyre.
#
# Why this is needed: the sentient/ install ships a prebuilt libspyre_comms.so
# compiled against an older flex. Two incompatibilities follow from the flex
# upgrade:
#   1. flex::createDmaParams gained a 9th parameter, so the old binary fails to
#      resolve it at import time ("undefined symbol: ...createDmaParams...").
#   2. flex's header cleanup dropped a transitive include chain that used to
#      supply senlib::v2::PinnedMemoryWrapper. spyre-comms #345 adds the direct
#      includes, so the spyre-comms checkout must be new enough to contain it.
#
# The staging directory exists because spyre-comms wants a single
# AIU_RUNTIME_INSTALL_DIR with include/ and lib/ subdirs, while the fresh flex
# has its headers in the source tree and its library in the build tree. The
# installed tree is copied first so non-flex headers (concurrentqueue, etc.)
# remain available, then the fresh flex headers are overlaid on top.
set -euo pipefail

FLEX_SRC="$HOME/dt-inductor/flex/flex/include"
FLEX_LIB="$HOME/dt-inductor/build/senbfcc/flex"
SENTIENT="$HOME/dt-inductor/sentient"
COMMS_SRC="$HOME/dt-inductor/spyre-comms"
COMMS_BUILD="$HOME/dt-inductor/build/spyre-comms"
STAGE="$HOME/dt-inductor/build/flex-stage"

echo "== checking spyre-comms has the senlib include fix (#345)"
if ! grep -rq "senlib::v2::PinnedMemoryWrapper" "$COMMS_SRC/src/"; then
  echo "   WARNING: could not find the expected senlib usage; check the checkout"
fi

echo "== staging an install-shaped fresh flex at $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE/include" "$STAGE/lib"
cp -r "$SENTIENT/runtime/include/." "$STAGE/include/"   # keeps concurrentqueue et al.
cp -r "$FLEX_SRC/." "$STAGE/include/"                   # fresh flex headers win
ln -sf "$FLEX_LIB/libflex.so" "$STAGE/lib/libflex.so"

echo "== backing up the current install"
if [[ -d "$SENTIENT/spyre_comms" && ! -d "$SENTIENT/spyre_comms.bak-aug19" ]]; then
  cp -a "$SENTIENT/spyre_comms" "$SENTIENT/spyre_comms.bak-aug19"
  echo "   -> $SENTIENT/spyre_comms.bak-aug19"
fi

echo "== configuring"
mkdir -p "$COMMS_BUILD"
cd "$COMMS_BUILD"
cmake "$COMMS_SRC" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_INSTALL_PREFIX="$SENTIENT/spyre_comms" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DAIU_RUNTIME_INSTALL_DIR="$STAGE" \
  -DBUILD_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF

echo "== building and installing"
make -j "${MAX_JOBS:-16}" install

echo "== verifying symbol compatibility with the fresh flex"
if nm -uC "$SENTIENT/spyre_comms/lib/libspyre_comms.so" | grep -q "createDmaParams.*bool, bool, bool)"; then
  echo "   OK: 9-arg createDmaParams (matches current flex)"
else
  echo "   WARNING: unexpected createDmaParams arity; check the flex staging"
fi

echo
echo "Done. Build torch-spyre with Multi-Spyre via:"
echo "  source scripts/kvc-build-env.sh && USE_SPYRE_CCL=1 python3 setup.py build_ext --inplace -j 16"
