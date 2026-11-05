# Persistent KV offload M4 test results

Branch: `kvc-offload-m4`  
Date: 2026-10-31

## Build under test

The eager rank-scoped migration used the isolated stack at
`/tmp/kvc-offload-m4-dev-stack-1000800000`:

- Flex came from
  `/home/yzhu/dt-inductor/flex-worktrees/persistent-shared-kv-pool-lifecycle`
  with its eager attachment changes and was installed under the stack's
  `runtime` directory.
- torch-spyre came from this worktree at baseline `a1c08b470` plus the
  uncommitted migration changes.
- torch-spyre rebuilt successfully against that Flex install with
  `USE_SPYRE_PROFILER=0`.
- Spyre Comms came from
  `/home/yzhu/dt-inductor/sentient/spyre_comms-m4-compat`.

Profiler support was disabled because the installed libaiupti headers do not
provide `AIUPTI_ACTIVITY_KIND_THREAD`. The shared-pool extension compiled,
linked, installed, and imported against the intended Flex library.

## Single-process shared-pool suite

Run from the repository root with the isolated runtime libraries first on
`LD_LIBRARY_PATH`:

```bash
python -m pytest \
  tests/test_shared_pool_lifecycle.py \
  tests/test_shared_host_pool.py \
  tests/test_shared_metadata.py \
  tests/test_kv_offload.py \
  tests/test_kv_page_validator.py -q
```

Result: **60 passed, 1 skipped, 3 warnings, 7 subtests passed** in 18.77
seconds.

The skip was the existing isolated-runtime guard. The warnings were existing
CPU fallback warnings from `torch.arange`. This suite covers eager rank
selection, aggregate alias close, independently retained metadata/data/read-pin
leases, persistent generation management, raw copies, page validation, and the
metadata claim/publish/lookup/pin/evict protocol.

The skipped fresh-parent case was also run alone after its cleanup assertion
was added: **1 passed** in 11.88 seconds and confirmed that the spawned helper
removes its generation.

## Two-card rank-scoped verification

Run with `SPYRE_DEVICES=0,1` and the same isolated libraries:

```bash
bash tests/run_test.sh \
  tests/configs/distributed_tests/test_shared_host_pool.yaml \
  tests/configs/distributed_tests/test_kv_offload.yaml \
  tests/configs/distributed_tests/test_shared_metadata.yaml
```

Each configuration launched two torchrun ranks and passed:

| Configuration | Result | Time |
|---|---:|---:|
| `test_shared_host_pool.yaml` | 1 passed | 8.47 s |
| `test_kv_offload.yaml` | 1 passed, 1 warning | 8.03 s |
| `test_shared_metadata.yaml` | 1 passed | 8.25 s |

The tests use one global metadata directory with separate rank-homogeneous
arenas. Each rank receives only `host-r<rank>` and performs DMA through that
eager local handle. The metadata test also covers peer metadata visibility,
same-key claim election, cross-process pin/eviction blocking, slot reuse, and
rank 1 alone keeping exact-generation destroy at `POOL_BUSY`.

## Repository gates and stale API scan

- `make check-all-configs` passed with 0 duplicate, missing, or unlabeled
  tests and 183 pre-existing dead-pattern warnings.
- Selected pre-commit hooks passed for the binding, type stub, helpers, and all
  single-process and distributed shared-pool tests.
- The obsolete-API scan found no consumer call to deferred data resolution,
  no `AttachResult.metadata`, and no metadata detach. Its only match was the
  intentional assertion that `SharedMetadata` has no `resolve_pool` attribute.

The broader `tests/test_stream.py tests/test_spyre.py` regression run passed
with **72 passed, 3 skipped, 5 expected failures, 1 warning, and 16 subtests**
in 21.73 seconds.
