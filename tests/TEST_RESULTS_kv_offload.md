# Persistent KV offload M4 test results

Branch: `kvc-offload-m4`  
Date: 2026-10-24

## Build under test

The final verification used a fresh isolated stack rooted at
`/tmp/kvc-offload-m4-final.5XKfSK`:

- Flex M4 commit `6c863b148`, built and installed as RelWithDebInfo.
- torch-spyre `kvc-offload-m4`, built against that install.
- `USE_SPYRE_PROFILER=0`, because profiler API migration is outside this
  shared-pool change.
- Spyre Comms from `/opt/ibm/spyre/spyre-comms`.

Both native builds completed. The torch-spyre build emitted existing compiler
warnings in tensor-layout and distributed code; it emitted no M4 shared-pool
build error.

## Focused lifecycle, M1, and M2 suite

Run from the repository root with the isolated Flex library first on
`LD_LIBRARY_PATH`:

```bash
python -m pytest \
  tests/test_shared_pool_lifecycle.py \
  tests/test_shared_host_pool.py \
  tests/test_kv_offload.py \
  tests/test_kv_page_validator.py \
  tests/test_shared_metadata.py -v
```

Result: **38 passed, 1 skipped, 3 subtests passed** in 6.40 seconds.

The skipped case was `test_blocked_evict_releases_gil`, whose explicit guard
skips when an earlier test has initialized the parent process's runtime. A
fresh process running only that test passed in 11.48 seconds.

This suite covers persistent generation management, descriptor pickling,
stale-generation rejection, attachment and resolved-child lease ownership,
host-pool geometry, raw tensor and KV-page copies, page validation, metadata
claim/publish/lookup/pin/evict, slot reuse, and stale metadata references.

## Two-card cross-process verification

Run from `tests/` with `SPYRE_DEVICES=0,1` and the same isolated
`LD_LIBRARY_PATH`:

```bash
SPYRE_DEVICES=0,1 bash run_test.sh \
  configs/distributed_tests/test_shared_host_pool.yaml
SPYRE_DEVICES=0,1 bash run_test.sh \
  configs/distributed_tests/test_kv_offload.yaml
SPYRE_DEVICES=0,1 bash run_test.sh \
  configs/distributed_tests/test_shared_metadata.yaml
```

Each configuration launched two torchrun ranks on two Spyre cards and passed
its one distributed test:

| Configuration | Result | Time |
|---|---:|---:|
| `test_shared_host_pool.yaml` | 1 passed | 8.56 s |
| `test_kv_offload.yaml` | 1 passed, 1 warning | 8.77 s |
| `test_shared_metadata.yaml` | 1 passed | 8.05 s |

Only rank 0 performed `create_pool`, `open_pool`, and `destroy_pool`; rank 1
only attached and detached. The metadata test exercised distinct-key writes,
same-key claim election, peer reload, read-pin eviction blocking, unrelated
progress while eviction waits, slot reuse/versioning, unpublished and
published visibility, and rank 1 alone keeping destruction busy.

## Repository and regression gates

```bash
make check-all-configs
python -m pytest tests/test_stream.py tests/test_spyre.py -v
```

- `make check-all-configs`: passed with 0 duplicate, missing, or unlabeled
  tests. It reported 183 pre-existing dead-pattern warnings.
- Stream/device regression suite: **72 passed, 3 skipped, 5 xfailed, 16
  subtests passed** in 29.10 seconds. The skips and expected failures are the
  repository's configured device/dtype cases.
