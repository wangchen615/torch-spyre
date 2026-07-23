# KV-Cache Offload — Implementation Plan (Epics, Sub-Issues, Milestones)

| Field | Value |
|---|---|
| Status | Planning |
| Created | 2026-07-23 |
| Project board | [torch-spyre project #2, view 23](https://github.com/orgs/torch-spyre/projects/2/views/23) |
| Label (all issues) | `kvc-offloading` |
| Issue home | **All issues are created in the `torch-spyre` repo** and added to the project board. |
| Milestone 1 due | **July 2026** |
| Milestone 2 due | **August 2026** |
| Design docs | torch-spyre `docs/source/architecture/raw_copy_kv_offload.md`; the hardware-runtime shared-host-pool RFC; the spyre-inference upstream-connector-port RFC |

> **Terminology.** This plan and every issue it describes use **neutral terms**. The
> per-process C++ device runtime is the **hardware runtime**. The cross-process shared
> host memory it provides is the **shared host memory pool** (a **secondary memory
> pool**). Issue titles and bodies must not name internal component or library
> code-names — say "hardware runtime" and "secondary / shared host memory pool"
> instead.

## 1. What this plan covers

Three design documents describe a three-layer KV-cache offload stack:

1. **Hardware-runtime layer** — owns the mechanism: a raw (byte-exact) device↔host
   DMA primitive (`copyRaw`), a DMA-able shared host memory pool
   (`SharedHostPool`) addressed by integer slot, and a shared directory
   (`SharedHostMetadata`, block-hash→slot with a concurrency protocol).
2. **torch-spyre layer** (this repo) — a thin Python surface: turn a
   `device("spyre")` tensor into the hardware-runtime device address, and expose
   the runtime's raw-copy + pool + directory objects to Python through
   `torch_spyre._C`.
3. **spyre-inference layer** — the vLLM connector and cache policy: a
   `SpyreOffloadingSpec` and handlers that ride the upstream `OffloadingConnector`.

This document turns those designs into a concrete, dependency-ordered issue backlog:
**3 epics (1 design + 2 milestone) + 18 sub-issues (3 design-doc + 15 implementation)
= 21 issues**, all created in `torch-spyre`, all labeled `kvc-offloading`, all added to
project board view 23. The three design docs are one deliverable, so they sit as three
sub-issues under a single **design epic** (one RFC-merge PR per layer).

## 2. Current code status (grounds the scoping)

Assessed against the latest `main` of each layer on 2026-07-23.

| Layer | Already on `main` | Missing — the actual work |
|---|---|---|
| **Hardware runtime** | `CompositeAddress` (with `total_size()`, `is_single_chunk()`, multi-chunk `chunks_`); `DmaParams` where **`dci == nullptr` is already a straight byte copy**; `createDmaParams(void*, size_t, bool, const CompositeAddress*, dci)`; `launchOperationH2D/D2H`; `fillAsync(CompositeAddress*)` | Public `copyRaw`; `SharedHostPool`; `SharedHostMetadata`; host-buffer registration. **The DMA engine and device-address model exist; the shared-pool building blocks do not.** |
| **torch-spyre** | `SpyreStream::copyAsync` / `copyAsyncImpl` — **already accepts `dci == nullptr` and sizes the DMA by `CompositeAddress::total_size()`**; `SharedOwnerCtx::composite_addr` used across `job_plan.cpp` / `spyre_ccl.cpp`; `copy_tensor` pybind; `getStreamFromPool` | `copy_tensor_raw` binding; `get_composite_address` accessor; `SharedHostPool` / `SharedHostMetadata` pybind passthroughs; `get_dma_stream` accessor |
| **spyre-inference** | `spyre_worker.py`, `spyre_model_runner.py`, `platform.py` | **No `kv_offload/` package, no connector, no `register_kv_caches` wiring** — greenfield connector work |

**Two consequences that shape the milestone split:**

- **The device-address model migration is already done.** torch-spyre `main` already
  resolves `ctx->composite_addr` everywhere. The M1 raw copy is therefore *not* blocked
  on any address-handle migration — `copyAsyncImpl` already does a `dci == nullptr`
  byte copy sized by `total_size()`. M1's torch-spyre work is small and low-risk.
- **M1 has almost no hardware-runtime dependency; M2 is where the runtime dependency
  lives.** M1 copies a device page into a process-local `torch` CPU tensor — the
  existing plumbing already supports this. Only M2's **shared** pool needs the three
  new runtime building blocks (`SharedHostPool`, `SharedHostMetadata`, and `copyRaw`
  into a pool slot).

## 3. Milestones and their boundaries

- **Milestone 1 — per-instance host-RAM offload (due July 2026).** A user runs
  `vllm serve … spec_name: SpyreOffloadingSpec` and gets byte-exact host-RAM offload
  that survives across requests. Each instance has its *own* host tier. Minimal
  hardware-runtime work; the copy is byte-exact via the raw path (not the converting
  `copy_tensor`, which drifts ~1 ULP and is a correctness defect for KV data).
- **Milestone 2 — cross-instance shared host pool (due August 2026).** The host tier
  becomes a single **shared host memory pool** provided by the hardware runtime and
  shared by co-located instances: a block offloaded by one instance is reloaded by
  another with one raw DMA and no serialization. Gated on the runtime's three building
  blocks landing.

A milestone spans multiple layers; **every individual issue is closeable within one
PR in one repo.** Cross-layer ordering is expressed as dependencies, not as multi-repo
issues.

## 4. Epics

Three epics. **The design work is one epic (E-DESIGN)** whose three sub-issues each
merge one RFC (one PR, one layer) — the three design docs are a single deliverable, not
three separate epics. The two milestone epics track the implementation sub-issues.

| ID | Title | Milestone | Closes with |
|---|---|---|---|
| **E-DESIGN** | `[Epic] KV-cache offload design docs (runtime + torch-spyre + spyre-inference)` | design | All three E-D* design sub-issues (§5.0) merged |
| **M1** | `[Epic] Milestone 1 — per-instance host-RAM KV offload end-to-end` | M1 (Jul 2026) | All M1-* sub-issues closed; M1 acceptance `vllm serve` green |
| **M2** | `[Epic] Milestone 2 — cross-instance shared host-memory KV pool` | M2 (Aug 2026) | All M2-* sub-issues closed; M2 acceptance (two-instance peer hit) green |

## 5. Sub-issue backlog

Every sub-issue below is one PR in one repo. **Layer** names the repo the PR lands in
(hardware-runtime / torch-spyre / spyre-inference). All issues are *filed* in
`torch-spyre` and labeled `kvc-offloading`; the "layer" tells the implementer which
codebase the PR targets.

### 5.0 Design epic (E-DESIGN) — 3 sub-issues

The three design docs are a single deliverable under **E-DESIGN**; each sub-issue is one
RFC-merge PR in one layer. They are independent of each other and of the milestone work.

#### E-D1 — merge the hardware-runtime shared host memory KV pool RFC

- **Layer:** hardware runtime
- **Epic:** E-DESIGN
- **Label:** `kvc-offloading`
- **Blocked by:** none
- **Description:** Merge the RFC that specifies the mechanism — the byte-exact raw copy
  primitive, the DMA-able shared host memory pool (slot-addressed), and the
  block-hash → slot directory with its concurrency protocol.

#### E-D2 — merge the torch-spyre Python-surface design doc

- **Layer:** torch-spyre
- **Epic:** E-DESIGN
- **Label:** `kvc-offloading`
- **Blocked by:** none
- **Description:** Merge `docs/source/architecture/raw_copy_kv_offload.md` (plus its
  figures) — the torch-spyre surface: the one tensor-aware address step and the thin
  bindings over the runtime's raw copy, pool, and directory.

#### E-D3 — merge the spyre-inference upstream-connector-port RFC

- **Layer:** spyre-inference
- **Epic:** E-DESIGN
- **Label:** `kvc-offloading`
- **Blocked by:** none
- **Description:** Merge the RFC (plus figures) that ports the upstream
  `OffloadingConnector` experience to spyre-inference — `SpyreOffloadingSpec`, handlers,
  and the M1/M2 milestone ladder.

### 5.1 Milestone 1 — 7 sub-issues

#### M1-F1 — hardware runtime: public raw (byte-exact) host↔device DMA

- **Layer:** hardware runtime
- **Milestone:** M1 (Jul 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** none
- **Blocks:** M1-T2 (optional route), M2-F1
- **Description:** Add a public raw-copy entrypoint on the runtime stream that performs
  a byte-exact host↔device DMA with **no dtype/layout conversion**. It wraps the
  existing straight-byte-copy path (the DMA params where the conversion descriptor is
  null) and the existing H2D/D2H launch. The copy length is the device allocation's
  physical size (`total_size()` — the padded/tiled byte count, **not**
  `numel × itemsize`); the call asserts the allocation is single-chunk. This is the
  named, testable primitive both milestones build on.
- **Acceptance:** a device page filled with a known pattern round-trips through a plain
  host buffer byte-for-byte; a multi-chunk allocation is rejected; length is derived
  from `total_size()`, never the logical byte count.

#### M1-T1 — torch-spyre: `get_composite_address` accessor

- **Layer:** torch-spyre
- **Milestone:** M1 (Jul 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** none
- **Blocks:** M1-T2
- **Description:** Add a read-only Python accessor returning an opaque handle over the
  device address that backs a `device("spyre")` tensor's storage (the address already
  held on the tensor's owner context — no new bookkeeping). This is the one
  tensor-aware step the design assigns to torch-spyre. Handle holds no ownership and is
  invalidated when the tensor's storage is freed. Bind next to the existing copy/stream
  bindings.
- **Acceptance:** returns a handle whose reported chunk shape matches the tensor's
  allocation; rejected after the tensor storage is freed; no change to existing copy
  paths.

#### M1-T2 — torch-spyre: `copy_tensor_raw(host_tensor, dev_tensor, …)` binding

- **Layer:** torch-spyre
- **Milestone:** M1 (Jul 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-T1 (optionally M1-F1 if routed through the runtime's new public
  raw copy; otherwise reuses the existing null-conversion copy path already on `main`)
- **Blocks:** M1-S1, M2-T2
- **Description:** Bind a byte-exact raw copy between a CPU `torch.Tensor` and a
  `device("spyre")` tensor. Resolve the device address via M1-T1 and issue the copy
  with a **null conversion descriptor** (the existing implementation already accepts
  this and sizes by `total_size()`), so it reproduces the device page's bytes exactly.
  `non_blocking=False` synchronizes after enqueue; `True` returns after enqueue and the
  caller synchronizes. This is the tensor-form of the primitive M1's connector uses;
  the slot-addressed form is M2-T2.
- **Acceptance:** allocate a device tensor with a known fp16 pattern, copy D2H into a
  CPU tensor, zero the device tensor, copy H2D back, assert byte-equal; assert the
  converting `copy_tensor` path is unaffected.

#### M1-T3 — torch-spyre: `get_dma_stream` accessor

- **Layer:** torch-spyre
- **Milestone:** M1 (Jul 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** none
- **Blocks:** none (used by S-layer as an optional dedicated stream)
- **Description:** Thin wrapper over the existing pooled-stream accessor so the
  connector can keep a dedicated DMA stream for offload/reload, letting it overlap the
  compute stream once an async path lands. Copy bindings accept an optional explicit
  stream; when omitted they use the current stream for the device.
- **Acceptance:** returns a usable stream handle for a valid device; copies issued on it
  complete correctly.

#### M1-S1 — spyre-inference: `SpyreKvDmaCopier` + `kv_offload` package scaffold

- **Layer:** spyre-inference
- **Milestone:** M1 (Jul 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-T2
- **Blocks:** M1-S2
- **Description:** Create the `spyre_inference/v1/kv_offload/` package (`__init__.py`)
  and `copier.py` — a thin, stateless wrapper around the byte-exact `copy_tensor_raw`
  binding exposing `copy_d2h` / `copy_h2d`. Neither method allocates. Bump the
  torch-spyre pin to one that exposes `copy_tensor_raw`. Add
  `test_copier_round_trip.py` (Spyre-gated): known fp16 pattern D2H → mutate host →
  H2D → assert content.
- **Acceptance:** round-trip test passes on a Spyre runner; the copier never allocates;
  the pin bump is the only dependency change.

#### M1-S2 — spyre-inference: `SpyreCpuOffloadingHandlers`

- **Layer:** spyre-inference
- **Milestone:** M1 (Jul 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-S1
- **Blocks:** M1-S3
- **Description:** Implement `handlers.py` — `SpyreCpuOffloadingHandlers` and a
  `_SingleDirectionSpyreHandler` implementing the upstream `OffloadingHandler` contract
  (`transfer_async` / `get_finished` / `shutdown`). Each direction walks block-id pairs
  and calls the copier. Host destinations are process-local pages built with
  `torch.zeros(...)` (the `pool` parameter is left `None` in M1 — it is the seam M2
  reuses). Add `test_handler_dispatch.py` (CPU-only): exercise both directions and
  assert content lands and `get_finished` reports success.
- **Acceptance:** handler dispatch test green on CPU runners; handler names mirror the
  upstream shape so the spec yields them unchanged.

#### M1-S3 — spyre-inference: `SpyreOffloadingSpec` + registration + M1 acceptance

- **Layer:** spyre-inference
- **Milestone:** M1 (Jul 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-S2
- **Blocks:** M2-S1
- **Description:** Implement `spec.py` — `SpyreOffloadingSpec` subclassing the upstream
  CPU offloading spec, overriding the handler-creation hook to return Spyre handlers and
  dropping the CUDA/XPU platform gate, inheriting the manager and block-count math. Add
  the lazy factory registration in `spyre_inference/__init__.py`. Add
  `test_spec_registration.py` (CPU-only). Run and record the M1 acceptance:
  `vllm serve … spec_name: SpyreOffloadingSpec` boots, reaches `register_kv_caches`
  without raising, reports a host-tier hit on a prefix-extending second prompt, and
  produces byte-identical `temperature=0` output vs. a no-offload baseline.
- **Acceptance:** registration resolves; the three M1 acceptance checks pass; no changes
  required to the Spyre worker or platform (verified by inspecting the diff).

### 5.2 Milestone 2 — 9 sub-issues

#### M2-F1 — hardware runtime: `SharedHostPool` (DMA-able shared host memory pool)

- **Layer:** hardware runtime
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-F1
- **Blocks:** M2-F4, M2-T1
- **Description:** A DMA-able shared host memory pool of fixed-size slots, index
  (slot-id) addressed. `create_or_attach(stream, name, num_slots, slot_bytes)`,
  `slot_count()`, `slot_bytes()`, internal `slot_ptr(i)` for the raw copy, and
  attach-refcount lifecycle (unlink on last-out). **Pinning is internal to the pool** —
  no raw host pointer crosses out of the runtime. This is the secondary memory pool the
  cross-instance milestone offloads into.
- **Acceptance:** two processes attaching the same named pool see the same slots; slot
  addressing is stable; pinning is handled internally with no external pointer exposure.

#### M2-F2 — hardware runtime: `SharedHostMetadata` (block-hash → slot directory)

- **Layer:** hardware runtime
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** none (parallel with M2-F1)
- **Blocks:** M2-F3, M2-T1
- **Description:** A shared directory mapping block-hash → slot-id with per-slot
  lifecycle state (empty → reserved → valid → empty). `lookup(hash) -> slot | miss`,
  `claim(hash) -> slot`, `publish(hash, slot)`, `evict(hash)`. Backed by the same shared
  segment discipline as the pool. This is the bookkeeping that makes cross-instance
  reuse possible.
- **Acceptance:** claim/publish/lookup/evict behave across two attached processes;
  lifecycle transitions are enforced.

#### M2-F3 — hardware runtime: concurrency protocol (locks + generation + publish gate)

- **Layer:** hardware runtime
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-F2
- **Blocks:** M2-F4
- **Description:** The correctness protocol over the directory: a process-shared
  directory lock, a per-slot read/write pin (read = reload, write = evict), and a
  per-slot generation counter for reuse/ABA detection, plus a publish-on-completion gate
  so a reader only ever observes a slot whose write has completed. A stale or mid-write
  slot must degrade to a **cache miss, never torn bytes**.
- **Acceptance:** under concurrent multi-process load, a slot reused mid-copy fails the
  generation check and is treated as a miss; no torn read is ever observable.

#### M2-F4 — hardware runtime: `copyRaw` into a pool slot (+ multi-chunk for 1p5)

- **Layer:** hardware runtime
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-F1, M2-F3
- **Description:** Extend the raw copy so its host endpoint is a pool slot (addressed by
  slot-id via `slot_ptr`) rather than an arbitrary host buffer, preserving the
  byte-identical-layout invariant. Add the multi-chunk (interleaved) chunk→domain
  descriptor path required on the platform where allocations may be multi-chunk, so the
  raw copy is correct beyond the single-chunk case. Cross-process round-trip test:
  store a page into a slot from one process, reload byte-for-byte from another.
- **Acceptance:** cross-process slot round-trip is byte-exact; the copy size is owned by
  the runtime (`total_size()`); multi-chunk allocations round-trip correctly.

#### M2-T1 — torch-spyre: `SharedHostPool` / `SharedHostMetadata` pybind passthroughs

- **Layer:** torch-spyre
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-F1, M2-F2
- **Blocks:** M2-T2, M2-S1
- **Description:** One-to-one pybind exposures of the runtime's pool and directory
  classes so spyre-inference reaches the whole mechanism through `torch_spyre._C`.
  torch-spyre adds nothing — no shared-segment creation, no locking, no directory logic.
  `slot_ptr` is intentionally **not** exposed to Python (the seam is the integer
  slot-id); the raw copy uses it in C++. Method sets track the runtime headers as a
  passthrough.
- **Acceptance:** create-or-attach from two processes see the same slots (driven from
  Python); no raw host pointer or device address crosses into Python.

#### M2-T2 — torch-spyre: slot-addressed `copy_tensor_raw(dev_tensor, pool, slot_id, …)`

- **Layer:** torch-spyre
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-T1, M1-T2, M2-F4
- **Blocks:** M2-S1
- **Description:** Add the slot-addressed overload: resolve the device tensor's address
  (M1-T1), then issue the runtime raw copy between the device page and the pool slot
  named by integer `slot_id`. The runtime owns the copy size, chunking, and the
  byte-identical-layout invariant; torch-spyre passes the address and the slot and
  computes no byte count. Add a Python round-trip test through a pool slot.
- **Acceptance:** device page → slot (offload) → fresh same-(shape,dtype) page (reload)
  is byte-exact from Python; raw host pointers never surface to Python.

#### M2-S1 — spyre-inference: `SpyreSharedOffloadingSpec` + registration

- **Layer:** spyre-inference
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-T1, M2-T2, M1-S3
- **Blocks:** M2-S2
- **Description:** Implement `shared_spec.py` — `SpyreSharedOffloadingSpec` subclassing
  M1's `SpyreOffloadingSpec`, reusing its handlers and copier **unchanged**. The only
  difference is the host destination: a slot in a shared host memory pool named by
  integer `slot_id` via a shared directory (block-hash → slot), instead of a
  process-local page. On store: `claim` a slot, D2H raw copy, then `publish` after the
  copy synchronizes. On load: `lookup` the hash, H2D raw copy from the slot. Add the
  third lazy factory registration; keep it inert when not selected and on builds without
  the M2 runtime surface.
- **Acceptance:** the spec resolves via the factory; importing the plugin on a build
  without the M2 surface does not error; M1 path unaffected.

#### M2-S2 — spyre-inference: shared-pool round-trip + torn-read test

- **Layer:** spyre-inference
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-S1
- **Blocks:** M2-S3
- **Description:** `test_shared_pool_round_trip.py` (Spyre-gated): store a known-pattern
  device page into a slot (`claim` + D2H raw copy + `publish`), then `lookup` + H2D raw
  copy into a fresh page and assert byte-exact content. Also assert a mid-write slot
  degrades to a miss via the directory gate (torn-read safety).
- **Acceptance:** byte-exact round-trip through a slot; a mid-write slot yields a miss,
  never torn bytes.

#### M2-S3 — spyre-inference: cross-instance test + M2 acceptance

- **Layer:** spyre-inference
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-S2
- **Description:** `test_cross_instance.py` (two-process): process A stores+publishes a
  block into the shared pool; process B, attaching the same named pool + directory,
  looks up the same content hash and reloads it — assert a cross-instance hit and
  byte-identical reload. Run and record the M2 acceptance: two `vllm serve` instances on
  one host with a shared pool, where the second instance gets a host-tier hit on a block
  the first offloaded, on its first request, with byte-identical `temperature=0` output.
- **Acceptance:** cross-instance peer hit works; both M2 acceptance runs pass; M1 path
  unaffected.

## 6. Dependency graph

```text
E-DESIGN epic (3 sub-issues, independent, one RFC-merge PR each):
  E-D1 (runtime RFC)   E-D2 (torch-spyre design)   E-D3 (spyre-inference RFC)

MILESTONE 1  (due Jul 2026)
  M1-F1 ─┐                         (runtime: public raw copy)
  M1-T1 ─┴─► M1-T2 ─► M1-S1 ─► M1-S2 ─► M1-S3
  M1-T3 (parallel; optional dedicated stream)

MILESTONE 2  (due Aug 2026)
  M1-F1 ─► M2-F1 ─┐
  M2-F2 ─► M2-F3 ─┴─► M2-F4 ─┐
  M2-F1, M2-F2 ─► M2-T1 ─────┼─► M2-T2 ─► M2-S1 ─► M2-S2 ─► M2-S3
  M1-T2 ─────────────────────┘           ▲
  M1-S3 ─────────────────────────────────┘
```

**Cross-layer seams** (dependencies that cross a repo boundary):

- `M1-T2` (torch-spyre) → `M1-S1` (spyre-inference): the raw-copy binding + pin bump
  gate the connector.
- `M2-F1` / `M2-F2` (runtime) → `M2-T1` (torch-spyre); `M2-F4` (runtime) → `M2-T2`
  (torch-spyre): runtime building blocks gate the torch-spyre bindings.
- `M2-T1` / `M2-T2` (torch-spyre) → `M2-S1` (spyre-inference): the torch-spyre surface
  gates the shared spec.

## 7. Issue counts

| Layer | Design subs | M1 subs | M2 subs |
|---|---|---|---|
| hardware runtime | 1 (E-D1) | 1 (M1-F1) | 4 (M2-F1…F4) |
| torch-spyre | 1 (E-D2) | 3 (M1-T1…T3) | 2 (M2-T1,T2) |
| spyre-inference | 1 (E-D3) | 3 (M1-S1…S3) | 3 (M2-S1…S3) |

Plus **3 epics**: E-DESIGN, M1, M2.

**Total: 21 issues** — 3 epics (1 design + 2 milestone) + 3 design subs + 7 M1 subs +
9 M2 subs (18 sub-issues). All created in `torch-spyre`, all labeled `kvc-offloading`,
all added to project board view 23.

## 8. Creation procedure (for when we proceed)

1. Create the 3 epics (E-DESIGN, M1, M2) in `torch-spyre` with the `kvc-offloading`
   label; spot-check.
2. Create the 18 sub-issues (3 design + 15 implementation), each referencing its epic
   (E-DESIGN for E-D*, the milestone epic for M1-*/M2-*) and listing its `Blocked by` /
   `Blocks` from §5 in the body (GitHub issues do not enforce dependencies natively —
   encode them as `Blocked by #NNN` lines and/or project-board relations).
3. Set the two milestones (M1 → July 2026, M2 → August 2026) and assign every sub-issue
   to its milestone.
4. Add all 21 issues to [project board view 23](https://github.com/orgs/torch-spyre/projects/2/views/23).
