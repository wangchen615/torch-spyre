# KV-Offload (`copy_tensor_raw`) — Test Results

Branch: `kvc-offload-poc`
Date: 2026-09-22

## Branch contents

Three stacked commits on current `main`:

1. `7b7c107b` — SharedPool/SharedHostPool pybind passthrough (squashed from upstream #3915)
2. `040cd8f7` — `copy_tensor_raw` pybind wrapper, `SpyreStream::copyRaw`, and KV-offload tests
3. `b51d31f8` — compat: call `flex::compositeAddressToDmva` for the installed flex

## How to run

```bash
python3 -m pytest tests/test_kv_offload.py -v
```

Undeclared test dependency: `transformers` (not in `pyproject.toml`) —
`uv pip install transformers` if missing. The two `test_real_model_*` tests
import `AutoConfig`.

## Summary: 4 passed, 2 failed (13.36 s, no hangs)

| Test | Result | Notes |
|---|---|---|
| `test_normal_copy_tensor_unaffected` | PASS | Regular tensor copy path is unaffected by the new raw path. |
| `test_page_view_offload` | PASS | Page-view (sliced) offload round-trip. |
| `test_real_model_large_slot` | PASS | Real model config, large host-pool slot. |
| `test_real_model_small_slot` | PASS | Real model config, small host-pool slot. |
| `test_kv_offload_reload_zeroed` | **FAIL** | Alignment (see below). |
| `test_kv_offload_reload_diff_tensor` | **FAIL** | Alignment (see below). |

## Why the two failures — pre-existing branch self-contradiction

Both failures are the **same** root cause, and both are a defect **in the
source branch (thakobian:copy_tensor_raw / upstream #3964)**, not a rebase
artifact and not a hardware issue:

- The branch's `copy_tensor_raw` implementation adds a strict 128-byte
  alignment check (flex `DEVICE_ALIGNMENT = 128`):

  ```
  RuntimeError: copy_tensor_raw: subrange must be 128-byte aligned,
                got offset=0 length=20
  ```

- But these two tests allocate `torch.randn(10, device="spyre", dtype=torch.float16)`
  = **10 fp16 elements = 20 bytes**, which is not 128-byte aligned. The tests
  were never updated after the alignment check was introduced, so they fail on
  the original branch too (verified: the test file and the alignment check are
  byte-identical to the branch tip).

The old pre-alignment-check code **hung** on this misaligned DMA (and wedged
the card, requiring a hot reset). The new code turns that hang into an
immediate, correct `RuntimeError` — this is the intended improvement.

### Fix (belongs upstream, in thakobian's branch)

Either:

- Align the test tensors, e.g. `torch.randn(64, ...)` (64 fp16 = 128 bytes), or
- Relax/remove the alignment check if sub-stick raw copies are meant to be
  supported.

Not fixed here to keep this branch a faithful rebase of the upstream work.

## Card recovery note

The Spyre card at `0000:ab:00.0` had been left in a `RAS::CBRB::ControlBlockSoftFail`
state by an earlier run of the *old* (no-alignment-check) code hanging on a
misaligned DMA. Recovered with a user-level chip-mode hot reset:

```bash
/opt/ibm/spyre/senlib/bin/aiu_dd2_hot_reset -t chip -d ab:00.0
```

(`-t chip` goes through the VFIO `/dev` node and does not need root, unlike
`-t linux` which writes the root-owned sysfs `reset`.) After the reset the
runtime starts cleanly and the 4 passing tests are green.
