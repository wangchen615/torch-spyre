<!--
Apache 2.0 — this file is documentation (issue-creation drafts), not source code.
Draft issue bodies for the KV-cache offload backlog, formatted to match the
team's epic convention (see torch-spyre#778 "Stream Engine (P3)").
REVIEW BEFORE CREATING ON GITHUB. Nothing here has been filed yet.
-->

# KV-Cache Offload — GitHub Issue Drafts (review before creating)

These are the **exact issue bodies** to create on `torch-spyre`, formatted after the
team epic convention in [torch-spyre#778](https://github.com/torch-spyre/torch-spyre/issues/778):

- Each **epic** opens with a dense `**Summary:**` paragraph, then groups its children under
  `### Section` headers as a `- [ ]` checklist. Once the child issues exist, each checklist
  line links the child (`- [ ] #NNN — …`); the checklist doubles as the tracked-issues list
  (#778 uses classic task-list tracking, not native sub-issues).
- Each **child issue** is self-contained: a one-line summary, a **What** list, a **Unit
  tests / Tests** list naming the *exact* cases, an **Acceptance** list, and a **Blocked by**
  line referencing sibling IDs.

Structure: **3 epics + 21 child issues = 24 issues** (one child, M2-T2, is a deferred backlog
item with no milestone; it is created and cross-linked but is not part of any milestone). All
filed in `torch-spyre`, labeled `kvc-offloading`, added to
[project board view 23](https://github.com/orgs/torch-spyre/projects/2/views/23).

> **Open items to settle before creation (need your call):**
>
> 1. **Label.** `kvc-offloading` does **not exist** in the repo yet — create it first
>    (`gh label create kvc-offloading`). The `epic` label already exists (apply it to the 3
>    epics).
> 2. **Milestones.** M1 targets **end of August 2026**, M2 targets **end of September 2026**
>    (pushed back one month from the original Jul/Aug plan as of 2026-07-29). The repo has
>    only quarter milestones (`2026 Q2/Q3/Q4`); both August and September 2026 fall in
>    **`2026 Q3`**, so the GitHub milestone field is `2026 Q3` for all milestoned issues, and
>    the text below records the finer end-of-month targets. **M2-T2 is a deferred backlog item
>    with no milestone.**
> 3. **Shared-pool-first (per review).** M1 builds the **shared** `SharedHostPool`
>    (cross-process, attach-by-name) directly — *not* a private pool later rewritten. Only pool
>    creation and cross-process acceptance-test coverage differ; the block-hash→slot directory
>    (M2-F1) and the per-slot concurrency protocol (M2-F2) stay in M2. M1 also gains three
>    **prerequisites** (M1-P1/P2/P3) that block all of M1.
> 4. **torch-spyre numbering (per review).** **M1-T2** is the `SharedHostPool` pybind
>    passthrough (incl. cross-process sharing); **M1-T3** is the canonical `copy_tensor_raw`
>    binding (formerly M1-T2). Downstream `Blocked by` lines reference **M1-T3** for the copy
>    binding.
> 5. **Neutral terminology.** Bodies use "hardware runtime" and "shared/secondary host
>    memory pool" — no internal code-names. (Class names like `SharedHostPool` /
>    `copy_tensor_raw` are the *public API surface names* from the design docs, kept because
>    developers implement against them; they are not internal code-names.)
> 6. **Cross-references.** `Blocked by` lines use the draft IDs (M1-F1, …). On creation,
>    replace with the real `#NNN` once each issue exists (create in dependency order, or
>    create all then edit in the numbers).

---

## EPIC 1 — E-DESIGN

**Title:** `[Epic] KV-cache offload design docs (runtime + torch-spyre + spyre-inference)`
**Labels:** `epic`, `kvc-offloading` · **Milestone:** design

**Summary:** Land the three design documents that specify a three-layer KV-cache offload
stack, as one coherent deliverable. The **hardware-runtime** RFC owns the mechanism (a
byte-exact raw device↔host DMA, a DMA-able shared host memory pool addressed by integer
slot, and a block-hash→slot directory with a concurrency protocol). The **torch-spyre**
design specifies the thin Python surface (`torch_spyre._C`): the one tensor-aware
address step and passthrough bindings. The **spyre-inference** RFC ports the upstream
vLLM `OffloadingConnector` experience and defines the M1/M2 milestone ladder. The three
docs must be **mutually consistent**: one canonical copy signature
`copy_tensor_raw(dev_tensor, pool, slot_id, to_device)`, one integer `slot_id` seam, one
ownership split — verified by a cross-doc consistency pass.

### Design documents

- [ ] **E-D1: Merge the hardware-runtime shared host memory KV pool RFC** (layer: hardware runtime)
- [ ] **E-D2: Merge the torch-spyre Python-surface design doc** (layer: torch-spyre)
- [ ] **E-D3: Merge the spyre-inference upstream-connector-port RFC** (layer: spyre-inference)

### Closes with

- [ ] All three RFCs merged.
- [ ] Cross-doc consistency note recorded: the `copy_tensor_raw(dev_tensor, pool, slot_id,
      to_device)` signature, the integer `slot_id` seam, and the "pinning internal to the
      pool / no host-buffer-registration binding" statement appear **identically** in all
      three docs.

---

## EPIC 2 — M1

**Title:** `[Epic] Milestone 1 — host-RAM KV offload end-to-end (shared pool)`
**Labels:** `epic`, `kvc-offloading` · **Milestone:** M1 (end Aug 2026)

**Summary:** Deliver **byte-exact** KV-cache offload end-to-end: a user runs
`vllm serve … spec_name: SpyreOffloadingSpec` and gets host-RAM offload that survives across
requests, offloading into a **shared** host memory pool. The device↔host copy is byte-exact
via the raw slot-addressed path `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` — never
the converting copy (which drifts ~1 ULP and is a correctness defect for KV data). Spans three
layers plus prerequisites: the hardware runtime adds the public raw copy primitive and the
**shared** pool; torch-spyre adds the tensor-address accessor, the shared-pool binding, and the
canonical copy binding; spyre-inference builds the production connector (copier → handlers →
spec).

**Why shared-pool-first:** M1 and M2 have a compact combined schedule, and deferring the shared
pool to M2 would mean a large, hard-to-review M2 rewrite. We must build the pool (M1-F2) anyway,
so we build it **shared from the start** — only pool creation (a named, shared segment) and its
cross-process acceptance tests differ from a single-instance pool; everything else is identical.
The block-hash→slot **directory** (M2-F1) and the per-slot **concurrency protocol** (M2-F2) stay
in M2; M1's shared pool uses externally-coordinated slot ids without the safe-concurrency layer.

**Prerequisites (per discussion with @frankeh):** before any offload code, M1 requires a
reproducible custom-built env, a **recomputation baseline** on the *latest* spyre-inference with
a **pinned** version, and a documented **host CPU buffer model** for the raw copy. An earlier
proof-of-concept connector (converting copy into a host tensor, on an old spyre-inference commit,
capped at ~4K-token prompts) is **discarded** — production code is built fresh to the latest
design.

### Prerequisites (block all of M1)

- [ ] **M1-P1: Env — custom-built flex + torch-spyre + spyre-inference (reproducible, pinned)**
- [ ] **M1-P2: spyre-inference recomputation baseline on latest code + pin a version**
- [ ] **M1-P3: Define the host CPU buffer/tensor model for raw copy (before M1-F1)**

### Hardware runtime

- [ ] **M1-F1: Public raw (byte-exact) host↔device DMA**
- [ ] **M1-F2: `SharedHostPool` — shared host memory pool (cross-process, attach-by-name)**

### torch-spyre

- [ ] **M1-T1: `get_composite_address` accessor**
- [ ] **M1-T2: `SharedHostPool` pybind passthrough (incl. cross-process sharing)**
- [ ] **M1-T3: `copy_tensor_raw(dev_tensor, pool, slot_id, …)` binding (canonical)**

### spyre-inference

- [ ] **M1-S1: `SpyreKvDmaCopier` + `kv_offload` package scaffold**
- [ ] **M1-S2: `SpyreCpuOffloadingHandlers`**
- [ ] **M1-S3: `SpyreOffloadingSpec` + registration + M1 acceptance & benchmark**

### Closes with (concrete end-to-end acceptance)

- [ ] Env + baseline in place: the pinned custom stack builds, and the **recomputation
      baseline** (no offload) runs on latest spyre-inference with its TTFT-vs-prompt-length
      curve recorded (M1-P1/P2).
- [ ] `vllm serve … spec_name: SpyreOffloadingSpec` boots and reaches
      `register_kv_caches` without raising.
- [ ] **Load/restore KVC from device HBM:** a device→host→device round-trip of a KV block
      is **byte-for-byte** identical (byte-exact raw copy).
- [ ] **Cross-process pool sharing:** a KV page written to a shared-pool slot by one process
      is reloaded byte-exact by a second process attaching the same named pool.
- [ ] A prefix-extending second prompt reports a host-tier hit (`N > 0` blocks loaded from
      host).
- [ ] **Generation accuracy with reloaded KVC:** with `temperature=0`, generated tokens
      are **byte-identical** to a no-offload baseline.
- [ ] **Performance benchmark recorded:** per-block offload (D2H) / reload (H2D) latency and
      throughput (GB/s), and TTFT reduction on a cache hit vs. the recomputation baseline.
- [ ] No source changes to the Spyre worker or platform (verified by diff).

---

## EPIC 3 — M2

**Title:** `[Epic] Milestone 2 — cross-instance shared host-memory KV pool`
**Labels:** `epic`, `kvc-offloading` · **Milestone:** M2 (end Sep 2026)

**Summary:** Make the host tier a single **shared host memory pool** shared by every
co-located Spyre instance, so a KV block offloaded by one instance is reloaded by another
with **one raw DMA and no serialization** — at memory speed, no disk. Reuses M1's shared
pool object and canonical copy path **unchanged**; adds the cross-instance layer: a
`SharedHostMetadata` block-hash→slot directory, a per-slot concurrency/generation protocol
with a publish-on-DMA-completion gate (a stale or mid-write slot degrades to a **cache
miss, never torn bytes**), the multi-chunk (1p5) raw-copy path, and the
`SpyreSharedOffloadingSpec` connector wiring. Gated on the directory, the concurrency
protocol, and the multi-chunk copy landing. M2 targets the correctness **baseline**; a
dedicated DMA stream (overlap optimization) is **deferred to the backlog** (M2-T2), not part
of M2.

### Hardware runtime

- [ ] **M2-F1: `SharedHostMetadata` — block-hash → slot directory**
- [ ] **M2-F2: Concurrency protocol (locks + generation + publish gate) — full race coverage**
- [ ] **M2-F3: `copyRaw` multi-chunk (1p5) + cross-process slot round-trip**

### torch-spyre

- [ ] **M2-T1: `SharedHostMetadata` (+ shared-pool attach) pybind passthroughs**

### spyre-inference

- [ ] **M2-S1: `SpyreSharedOffloadingSpec` + registration**
- [ ] **M2-S2: Shared-pool round-trip + connector miss→recompute test**
- [ ] **M2-S3: Cross-instance test + M2 acceptance & benchmark**

### Closes with (concrete end-to-end acceptance)

- [ ] **Cross-instance peer hit:** two `vllm serve` instances on one host attach the same
      shared pool; the second gets a host-tier hit **on its first request** on a block the
      first offloaded — served by a device←host DMA, no recompute, no disk.
- [ ] **Generation accuracy with peer-reloaded KVC:** with `temperature=0`, B's tokens are
      **byte-identical** to a no-cache baseline.
- [ ] **Full data-race coverage** passes (the M2-F2 matrix) — zero torn reads, zero
      double-allocations.
- [ ] **Performance benchmark recorded:** cross-instance peer-hit latency (B's TTFT on a
      shared-pool hit) vs. full recompute, and shared-pool reload throughput vs. M1.
- [ ] M1 path unaffected.

### Deferred (backlog — not gating M2)

- **M2-T2: `get_dma_stream` dedicated DMA stream (overlap optimization).** Pulled out of M2 per
  review; schedule after the M2 baseline lands. **No milestone.**

---

# CHILD ISSUES

Each block below is one GitHub issue body. Title line and metadata are shown above the
`---` body separator; copy everything under **Body** into the issue.

---

## E-D1 — Merge the hardware-runtime shared host memory KV pool RFC

**Title:** `[E-D1] Merge the hardware-runtime shared host memory KV pool RFC`
**Labels:** `kvc-offloading` · **Milestone:** design · **Epic:** #\<E-DESIGN>

**Body:**

**Summary:** Merge the RFC that specifies the KV-offload **mechanism** at the hardware-runtime
layer: a byte-exact raw device↔host DMA primitive, a DMA-able shared host memory pool
(slot-addressed, pinned internally per IOMMU Function), and a block-hash→slot directory with
its concurrency protocol. This is the bottom layer the other two design docs consume.

**What**

- Land the RFC document in the hardware-runtime repo (RFC + any figures).
- Public API surface defined: `copyRaw(host_addr, CompositeAddress*, to_device)`,
  `SharedHostPool::create_or_attach(stream, name, num_slots, slot_bytes)`,
  `SharedHostMetadata::create_or_attach(...)`, the concurrency protocol, and the multi-chunk
  pack/rebuild contract.
- Ownership split stated: the hardware runtime owns the mechanism; torch-spyre owns only the
  tensor→address step and thin bindings; spyre-inference owns cache policy.

**Acceptance**

- [ ] RFC merged.
- [ ] API signatures match what E-D2 (torch-spyre) and E-D3 (spyre-inference) consume — one
      canonical `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` seam, integer
      `slot_id`, no caller-owned host-buffer path.

**Blocked by:** none

---

## E-D2 — Merge the torch-spyre Python-surface design doc

**Title:** `[E-D2] Merge the torch-spyre Python-surface design doc`
**Labels:** `kvc-offloading` · **Milestone:** design · **Epic:** #\<E-DESIGN>

**Body:**

**Summary:** Merge `docs/source/architecture/raw_copy_kv_offload.md` (plus figures) — the
torch-spyre Python surface: the one tensor-aware address step (`get_composite_address`) and
thin passthrough bindings over the hardware runtime's raw copy, pool, and directory.

**What**

- Land the design doc and its call-path figure.
- Specify a **single** `copy_tensor_raw` signature: `(dev_tensor, pool, slot_id, to_device,
  non_blocking)` — **no host-tensor overload**; the host destination is always a pool slot.
- State the invariants: pinning is internal to the pool (no host-buffer-registration
  binding); `slot_ptr` is not exposed to Python; raw host pointers never cross into Python.

**Acceptance**

- [ ] Doc merged.
- [ ] `copy_tensor_raw` signature, `slot_id` seam, and "pinning internal / no
      host-buffer-registration" statements are identical to the hardware-runtime RFC (E-D1).

**Blocked by:** none

---

## E-D3 — Merge the spyre-inference upstream-connector-port RFC

**Title:** `[E-D3] Merge the spyre-inference upstream-connector-port RFC`
**Labels:** `kvc-offloading` · **Milestone:** design · **Epic:** #\<E-DESIGN>

**Body:**

**Summary:** Merge the RFC (plus figures) that ports the upstream vLLM `OffloadingConnector`
experience to spyre-inference — `SpyreOffloadingSpec`, handlers, and the M1/M2 milestone
ladder.

**What**

- Land the RFC and its architecture figures.
- M1 copier must call the **canonical** `copy_tensor_raw(dev_tensor, pool, slot_id,
  to_device)` into the **shared** pool — not a converting copy into a host tensor.
- M2 differs from M1 only by adding the shared block-hash→slot directory + concurrency on top
  of the same shared pool (same copy signature).

**Acceptance**

- [ ] Doc merged.
- [ ] The M1 **and** M2 copier code both use the one canonical `copy_tensor_raw` signature
      (the "reuses M1 unchanged" claim is literally true — only the directory/concurrency layer
      is added in M2).
- [ ] No `copy_tensor_raw(dev_tensor, host_tensor, …)` host-tensor form remains in the doc.

**Blocked by:** none

---

## M1-P1 — Env: custom-built flex + torch-spyre + spyre-inference (reproducible, pinned)

**Title:** `[M1-P1] Prereq: reproducible custom-built flex + torch-spyre + spyre-inference env`
**Labels:** `kvc-offloading` · **Milestone:** M1 (end Aug 2026) · **Epic:** #\<M1>

**Body:**

**Summary:** Stand up a reproducible development/test environment with **custom builds** of all
three layers — the hardware runtime (flex), torch-spyre, and spyre-inference — pinned together
so KV-offload work is developed and tested against a known-good stack. This is a prerequisite
for *everything* in M1: no raw-copy, pool, or connector work can be verified without it.

**What**

- A documented, repeatable build of the three layers from source (custom-build flags for the
  hardware runtime, a torch-spyre build against that runtime, and a spyre-inference checkout
  wired to that torch-spyre).
- A single command / script (or CI recipe) that produces the environment from scratch on a
  Spyre-capable host.
- Recorded versions/commits of all three layers so the stack is reproducible.

**Tests / verification**

- From a clean environment, the build script produces a working stack: `import torch_spyre`
  succeeds, a trivial `device("spyre")` tensor op runs, and `vllm serve` (spyre-inference)
  starts against it.
- The recorded commit triple can be re-checked-out and rebuilt to the same working state.

**Acceptance**

- [ ] Documented, repeatable build of custom flex + torch-spyre + spyre-inference on a Spyre
      host.
- [ ] Recorded commit/version triple; a clean rebuild reproduces a working stack.
- [ ] A trivial spyre op and a `vllm serve` boot both succeed on the built stack.

**Blocked by:** none

---

## M1-P2 — spyre-inference recomputation baseline on latest code + pin a version

**Title:** `[M1-P2] Prereq: spyre-inference recomputation baseline on latest code + pin a version`
**Labels:** `kvc-offloading` · **Milestone:** M1 (end Aug 2026) · **Epic:** #\<M1>

**Body:**

**Summary:** Establish a **recomputation baseline** on the **latest** spyre-inference (no KV
offload) and **pin a spyre-inference version** for KV-offload development. Per discussion with
@frankeh: measure the baseline (recomputation, no offload) on current code *before* building
offload, so the offload numbers (TTFT vs. recompute) have a valid reference and so we know the
stack is stable at the prompt lengths we care about.

**Context / motivation**

- The earlier PoC ran on an **old spyre-inference commit** and could only reach **~4K-token
  prompts** despite runtime-config tuning; the target model
  `ibm-ai-platform/micro-g3.3-8b-instruct-1b` has a **32K** max length.
- We are building the production connector fresh, so the baseline must be taken on the
  **latest** spyre-inference, not the PoC commit.
- Goal: reach (or characterize the ceiling toward) **max model length without KV offload** on
  latest code, so offload work starts from a known-stable, known-fast baseline.

**What**

- Run recomputation (no-offload) generation on latest spyre-inference with
  `ibm-ai-platform/micro-g3.3-8b-instruct-1b`.
- **Explore proper prompt length and setup:** sweep prompt lengths (e.g. 1K, 2K, 4K, 8K, 16K,
  up toward 32K), record which lengths run cleanly and where/why it breaks, and capture the
  runtime configs tried.
- Record **baseline TTFT vs. prompt length** (recomputation) as the reference for later offload
  comparisons.
- **Pin a spyre-inference version/commit** (together with the M1-P1 flex/torch-spyre pins) as
  the development baseline for the KV-offload work.

**Tests / verification**

- Baseline generation runs on latest spyre-inference for the documented prompt-length set;
  results (max stable length, TTFT curve, configs) recorded in the issue/PR.
- The pinned spyre-inference commit is recorded and reproducible via the M1-P1 environment.

**Acceptance**

- [ ] Recomputation baseline (no offload) runs on **latest** spyre-inference with the target
      model; TTFT-vs-prompt-length curve recorded.
- [ ] Max stable prompt length characterized (target: push toward the 32K model max; document
      the ceiling and the blocking cause if not reached).
- [ ] A spyre-inference commit is **pinned** as the development baseline.

**Blocked by:** M1-P1

---

## M1-P3 — Define the host CPU buffer/tensor model for raw copy

**Title:** `[M1-P3] Prereq: define the host CPU buffer/tensor model for raw copy`
**Labels:** `kvc-offloading` · **Milestone:** M1 (end Aug 2026) · **Epic:** #\<M1>

**Body:**

**Summary:** Decide and document **how host CPU buffers/tensors are created for the raw copy** —
the host side of every `copyRaw`. This must be settled **before M1-F1**, because M1-F1's DMA and
its tests are written against this buffer model, and M1-F2's shared pool materializes slots the
same way. It defines the concrete mapping between a KV page and the host memory region it lands
in.

**What**

- Specify the host-buffer model the raw copy targets, covering:
  - **Regular contiguous CPU tensor** — a `torch.empty(total_size_bytes, dtype=torch.uint8)`
    (or equivalent) whose `data_ptr()` is the `host_addr`.
  - **Tensor over a pre-allocated buffer at an offset** — a KV page addressed inside a larger
    allocation at `base_ptr + slot_offset` (the shared-pool slot layout of M1-F2), so the host
    side is an arbitrary offset region, not only an allocation's start.
  - **Pinned vs. pageable** — whether the raw copy requires pinned/page-locked memory or works
    with pageable host memory (and where pinning happens — internal to the pool per M1-F2).
- Define how a device tensor's `total_size()` maps to the host byte region (contiguous and
  padded/tiled cases), so callers know the exact byte count and layout.
- Document the chosen model so M1-F1, M1-F2, and the torch-spyre bindings all target the same
  host-buffer contract.

**Tests / verification**

- A short reference snippet/spec showing each host-buffer form and the `host_addr`/size it
  yields.
- Reviewed and agreed by the runtime + torch-spyre owners (this is a design/contract issue; the
  executable tests live in M1-F1 and M1-F2).

**Acceptance**

- [ ] Host-buffer/tensor model documented: regular CPU tensor, offset-into-pre-allocated buffer,
      and the pinned/pageable decision.
- [ ] `total_size()` → host-region mapping specified for contiguous and padded/tiled tensors.
- [ ] M1-F1 and M1-F2 test plans reference this model (no divergent buffer assumptions).

**Blocked by:** M1-P1

---

## M1-F1 — Public raw (byte-exact) host↔device DMA

**Title:** `[M1-F1] Hardware runtime: public raw (byte-exact) host↔device DMA`
**Labels:** `kvc-offloading` · **Milestone:** M1 (end Aug 2026) · **Epic:** #\<M1>

**Body:**

**Summary:** Add a public raw-copy entrypoint on the hardware-runtime stream that performs a
byte-exact host↔device DMA with **no dtype/layout conversion**. It is the named, testable
primitive both milestones build on.

**What**

- New public method `copyRaw(host_addr, CompositeAddress*, to_device)` wrapping the existing
  straight-byte-copy path (DMA params with a **null** conversion descriptor) and the existing
  H2D/D2H launch.
- Copy length is the device allocation's physical size (`total_size()` — the padded/tiled
  byte count, **not** `numel × itemsize`).
- Single-chunk is the common case; the multi-chunk (1p5) path is **M2-F3**, not here.

**Unit tests**

The host side of the DMA is a raw byte region of exactly `total_size()` bytes. Cover the
distinct ways that region is materialized, and cover realistic sizes — **not** just a small
buffer:

- **Host buffer type — cover each of:**
  - **Regular contiguous CPU tensor** (`torch.empty(total_size_bytes, dtype=torch.uint8)`),
    passing its `data_ptr()` as `host_addr`. The simplest destination and the baseline case.
  - **Tensor viewed over a pre-allocated host buffer** — a single large host allocation
    (`bytearray` / `numpy` / a pinned block) with a KV page addressed at an **offset** inside
    it (`base_ptr + slot_offset`), i.e. the pool-slot layout M1-F2 uses. Verifies `copyRaw`
    copies to/from an arbitrary offset region, not only the start of an allocation.
  - **(If applicable) pinned/page-locked host buffer** vs. pageable — same bytes land either
    way; asserts `copyRaw` does not silently require pinned memory.
- **Tensor→region mapping — cover each of:**
  - A **contiguous** device tensor (single chunk).
  - A **padded/tiled** device tensor where `total_size()` > `numel × itemsize` (stick
    padding) → assert the copy length is `total_size()`, and the padding bytes round-trip too.
- **Size — cover small *and* large:**
  - A small page (a few sticks) for fast CI.
  - A **realistic KV-page size** and at least one **large** multi-MB allocation, so the DMA
    is exercised beyond a single descriptor / trivial length. Round-trip must stay
    byte-for-byte at scale.
- **Round-trip:** every case fills the device page with a known pattern → DMA to the host
  region (D2H) → zero the device page → DMA back (H2D) → assert **byte-for-byte** equal.
- The existing converting-copy path is untouched (regression).

**Acceptance**

- [ ] Byte-exact device→host→device round-trip passes for **each host-buffer type** (regular
      CPU tensor, offset-into-pre-allocated buffer, and pinned if applicable).
- [ ] Passes for both a contiguous and a padded/tiled tensor; copy length is `total_size()`
      (asserted *not* to be `numel × itemsize` for the padded case, and padding bytes
      round-trip).
- [ ] Passes at a **realistic KV-page size and at least one large multi-MB buffer**, not only
      a small buffer.
- [ ] Converting copy path unaffected.

**Blocked by:** M1-P3 — the host CPU buffer/tensor model must be defined first.

---

## M1-F2 — `SharedHostPool` (shared host memory pool, cross-process)

**Title:** `[M1-F2] Hardware runtime: SharedHostPool (shared host memory pool, cross-process)`
**Labels:** `kvc-offloading` · **Milestone:** M1 (end Aug 2026) · **Epic:** #\<M1>

**Body:**

**Summary:** A DMA-able **shared** host memory pool of fixed-size slots, index (slot-id)
addressed, **attachable by name across processes**. We build the **shared** pool directly in M1
(not a private pool we later rewrite): M1 and M2 have a compact combined schedule, and starting
shared avoids a large, hard-to-review M2 rewrite. Only **how the pool is created** (a shared,
named segment) and its **acceptance-test coverage** (cross-process) differ from a single-instance
pool — everything else (slot addressing, internal pinning, `slot_ptr` for the raw copy) is
identical. The block-hash→slot **directory** and the per-slot **concurrency protocol** stay in
M2 (M2-F1 / M2-F2); M1-F2 shares only the **pool** (slots + attach-by-name).

**What**

- `create_or_attach(stream, name, num_slots, slot_bytes)` backed by a **shared, named** segment
  (multiple processes attaching the same `name` map the same slots).
- `slot_count()`, `slot_bytes()`; internal `slot_ptr(i)` used by `copyRaw` — **not** exposed
  outside the hardware runtime.
- Attach-refcount lifecycle (unlink on last-out) across processes.
- **Pinning is internal to the pool** — pinned once per IOMMU Function inside `create_or_attach`;
  no raw host pointer crosses out of the hardware runtime.
- **Not in scope here:** the block-hash→slot directory (M2-F1) and locking/generation/publish
  protocol (M2-F2). M1 uses the shared pool with simple, externally-coordinated slot ids; the
  safe-concurrency layer is M2.

**Unit tests**

- **Single-process round-trip:** offload a device KV page into a slot, reload it →
  **byte-for-byte** equal.
- **Cross-process sharing (required — the pool is shared):**
  - Process A `create_or_attach(name, …)` and writes a known pattern into slot `i` (D2H);
    process B `create_or_attach(name, …)` attaches the **same** segment and reads slot `i` back
    (H2D) → **byte-for-byte** equal. Verifies the two processes map the same slots.
  - Slot addressing is consistent across processes (slot `i` is the same region in A and B).
  - Attach-refcount lifecycle across processes (segment survives while any process is attached;
    unlinks on last-out).
- Slot addressing is stable across a process's lifetime.
- No external host-pointer exposure (no way to obtain a raw slot pointer from outside the
  hardware runtime).

**Acceptance**

- [ ] Single-process offload→reload through a slot is byte-exact.
- [ ] **Cross-process:** a page written to slot `i` by one process is reloaded byte-exact by a
      second process attaching the same named pool.
- [ ] Consistent slot addressing across processes; attach-refcount lifecycle correct.
- [ ] Internal pinning; no external pointer exposure.

**Blocked by:** M1-F1, M1-P3

---

## M1-T1 — `get_composite_address` accessor

**Title:** `[M1-T1] torch-spyre: get_composite_address accessor`
**Labels:** `kvc-offloading` · **Milestone:** M1 (end Aug 2026) · **Epic:** #\<M1>

**Body:**

**Summary:** Add a read-only Python accessor returning an opaque handle over the device
address that backs a `device("spyre")` tensor's storage (already held on the tensor's owner
context — no new bookkeeping). This is the one tensor-aware step the design assigns to
torch-spyre.

**What**

- `get_composite_address(dev_tensor) -> CompositeAddressHandle` bound next to the existing
  copy/stream bindings.
- Handle holds no ownership; invalidated when the tensor's storage is freed.

**Unit tests**

- Returns a handle whose reported chunk shape matches the tensor's allocation.
- Rejected (raises) after the tensor's storage is freed.
- No change to existing copy paths (regression).

**Acceptance**

- [ ] Handle chunk shape matches allocation.
- [ ] Use-after-free rejected.
- [ ] Existing copy paths unaffected.

**Blocked by:** none

---

## M1-T2 — `SharedHostPool` pybind passthrough (incl. cross-process sharing)

**Title:** `[M1-T2] torch-spyre: SharedHostPool pybind passthrough (incl. cross-process sharing)`
**Labels:** `kvc-offloading` · **Milestone:** M1 (end Aug 2026) · **Epic:** #\<M1>

**Body:**

**Summary:** Expose the hardware runtime's **shared** `SharedHostPool` to Python via a pybind
passthrough, **including the cross-process sharing path** (attach-by-name). Because M1-F2 builds
the shared pool directly, its Python binding must be usable from two processes attaching the same
named pool — so the connector can drive the shared pool from Python in M1. torch-spyre adds
nothing: no shared-segment creation, no locking, no directory logic.

**What**

- Pybind exposure of `SharedHostPool.create_or_attach(stream, name, num_slots, slot_bytes)`,
  `slot_count()`, `slot_bytes()`.
- The **attach-by-name / shared** path works from Python (two Python processes attaching the
  same `name` see the same slots).
- `slot_ptr` stays **unexposed** to Python (the seam is the integer slot-id). The method set
  tracks the hardware-runtime header as a passthrough.
- No directory (`SharedHostMetadata`) here — that binding is M2-T1; M1 exposes only the pool.

**Unit tests**

- Create-or-attach a pool from Python and read back `slot_count()` / `slot_bytes()`.
- **Cross-process from Python (required):** two Python processes `create_or_attach` the same
  `name` → observe the **same slots** (a page written via the copy binding by one process is
  visible to the other). Driven end-to-end with M1-T3's `copy_tensor_raw`.
- No raw host pointer or device address crosses into Python.
- **Mimic real KV-cache offload data movement (required):** do **not** size the pool with
  arbitrary buffer lengths. Derive `slot_bytes` and `num_slots` from a **real model's** KV
  geometry so the test exercises production-shaped slots:
  - Grab dimensions from the target model (`ibm-ai-platform/micro-g3.3-8b-instruct-1b`):
    number of KV heads, head dim, block/page size, and layer count — compute a per-block KV
    page's `total_size()` (padded/tiled byte count) and use that as `slot_bytes`; use a
    realistic block count as `num_slots`.
  - Move a **real device KV page** of that shape through the pool via `copy_tensor_raw`
    (D2H → H2D), single-process **and** cross-process, and assert byte-for-byte equality — so
    the binding is verified against the actual offload payload, not a toy buffer.
  - Cover at least one **large** (multi-MB) slot size representative of a full-length prompt's
    per-block KV, not only the smallest page.

**Acceptance**

- [ ] Shared pool create-or-attach works from Python; `slot_count`/`slot_bytes` correct.
- [ ] **Two Python processes attaching the same named pool see the same slots** (cross-process
      sharing verified from Python).
- [ ] `slot_ptr` not exposed; no pointer/address leaks to Python.
- [ ] **Slot geometry is derived from a real model's KV dimensions** (not arbitrary sizes),
      and a real-shaped KV page round-trips byte-exact through the pool (single- and
      cross-process), including at least one large multi-MB slot.

**Blocked by:** M1-F2

---

## M1-T3 — `copy_tensor_raw(dev_tensor, pool, slot_id, …)` binding (canonical)

**Title:** `[M1-T3] torch-spyre: copy_tensor_raw(dev_tensor, pool, slot_id, …) binding (canonical)`
**Labels:** `kvc-offloading` · **Milestone:** M1 (end Aug 2026) · **Epic:** #\<M1>

**Body:**

**Summary:** Bind the **single canonical** byte-exact raw copy:
`copy_tensor_raw(dev_tensor, pool, slot_id, to_device, non_blocking=False)`. The host
destination is always a pool slot (the M1 shared pool; unchanged in M2) — **no host-tensor
overload** — which is what makes "M2 reuses M1's copy path unchanged" literally true.

**What**

- Resolve the device address via `get_composite_address` (M1-T1).
- Obtain the slot's host address from the pool **inside** the call (`pool.slot_ptr(slot_id)`),
  never surfaced to Python.
- Issue the hardware-runtime raw copy (M1-F1) with a **null conversion descriptor**; the runtime
  owns the copy size (`total_size()`), chunking, and the byte-identical-layout invariant —
  torch-spyre computes no byte count.
- `non_blocking=False` synchronizes after enqueue; `True` returns after enqueue and the caller
  synchronizes.

**Unit tests**

- Allocate a device tensor with a known fp16 pattern → offload D2H into a shared-pool slot →
  zero the device tensor → reload H2D → **byte-equal**.
- Reload the slot into a **different** same-`(shape, dtype)` tensor → **byte-equal**.
- **Cross-process:** offload into slot `i` from one process, reload from another process
  attaching the same named pool (via M1-T2) → **byte-equal**.
- The converting `copy_tensor` path is unaffected (regression).
- No raw host pointer surfaces to Python (API offers no slot-pointer accessor).

**Acceptance**

- [ ] Byte-exact round-trip both into the original and a fresh same-shaped tensor.
- [ ] Byte-exact cross-process round-trip through a shared-pool slot.
- [ ] Converting copy unaffected; no host pointer exposed.

**Blocked by:** M1-T1, M1-T2, M1-F1, M1-F2

---

## M1-S1 — `SpyreKvDmaCopier` + `kv_offload` package scaffold

**Title:** `[M1-S1] spyre-inference: SpyreKvDmaCopier + kv_offload package scaffold`
**Labels:** `kvc-offloading` · **Milestone:** M1 (end Aug 2026) · **Epic:** #\<M1>

**Body:**

**Summary:** Build the **production** `spyre_inference/v1/kv_offload/` package fresh to the
latest design (the proof-of-concept copier is discarded). `SpyreKvDmaCopier` is a thin, stateless
wrapper around the canonical `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` — byte-exact,
no converting copy, no host-tensor destination.

**What**

- New files: `__init__.py`, `copier.py`.
- `SpyreKvDmaCopier.copy_d2h` / `copy_h2d` forward to `copy_tensor_raw` against a pool
  `slot_id`. Neither method allocates.
- Bump the torch-spyre pin to one exposing `copy_tensor_raw` + `SharedHostPool`.

**Tests**

- `test_copier_round_trip.py` (Spyre-gated): known fp16 pattern → offload into a **shared-pool**
  slot (the M1-F2 shared `SharedHostPool`, attached by name) → mutate device page → reload →
  assert **byte-exact** content.
- Assert the copier never allocates (destinations are pre-provided pool slots).
- **Large / realistic tensor (required):** run the round-trip on a **real-model-shaped** KV page
  (slot sized from the target model's KV geometry, `ibm-ai-platform/micro-g3.3-8b-instruct-1b`),
  including at least one **large multi-MB** page — not only a small buffer — and assert it stays
  byte-exact at scale.

**Acceptance**

- [ ] Round-trip test passes on a Spyre runner and is **byte-exact** (not tolerance-based),
      offloading into a **shared-pool** slot (M1-F2 shared pool).
- [ ] **Byte-exact at a large / realistic KV-page size** (real model geometry, incl. a multi-MB
      page), not only a small buffer.
- [ ] Copier never allocates.
- [ ] The pin bump is the only dependency change.

**Blocked by:** M1-T3

---

## M1-S2 — `SpyreCpuOffloadingHandlers`

**Title:** `[M1-S2] spyre-inference: SpyreCpuOffloadingHandlers`
**Labels:** `kvc-offloading` · **Milestone:** M1 (end Aug 2026) · **Epic:** #\<M1>

**Body:**

**Summary:** Implement `handlers.py` — `SpyreCpuOffloadingHandlers` and a
`_SingleDirectionSpyreHandler` implementing the upstream `OffloadingHandler` contract, driving
the copier against pool slots.

**What**

- `handlers.py` with `gpu_to_cpu_handler` (store: device → host slot) and `cpu_to_gpu_handler`
  (load: host slot → device), names mirroring the upstream shape so the spec yields them
  unchanged.
- Each direction implements `transfer_async(job_id, transfer_spec)`, `get_finished()`,
  `shutdown()`; walks block-id pairs and calls `copier.copy_{d2h,h2d}` against a `slot_id`.
- Host destinations are slots in the **shared** `SharedHostPool` (M1-F2, attached by name); the
  `pool` is always present (no host-tensor path). M2 adds the block-hash → slot directory +
  concurrency on top of the same pool — the handler signature does not change.

**Tests**

- `test_handler_dispatch.py` (Spyre-gated): exercise both directions against block-id specs;
  assert content lands **byte-exact** and `get_finished` reports success.

**Acceptance**

- [ ] Handler dispatch test green; content round-trips byte-exact through the shared pool.
- [ ] Handler attribute names mirror upstream so the spec yields them unchanged.

**Blocked by:** M1-S1

---

## M1-S3 — `SpyreOffloadingSpec` + registration + M1 acceptance & benchmark

**Title:** `[M1-S3] spyre-inference: SpyreOffloadingSpec + registration + M1 acceptance & benchmark`
**Labels:** `kvc-offloading` · **Milestone:** M1 (end Aug 2026) · **Epic:** #\<M1>

**Body:**

**Summary:** Implement `spec.py` — `SpyreOffloadingSpec` subclassing the upstream CPU
offloading spec — register it, and run + record the M1 end-to-end acceptance and a
performance benchmark.

**What**

- `spec.py`: subclass the upstream CPU offloading spec; override the handler-creation hook to
  return Spyre handlers **and** create/attach the **shared** `SharedHostPool` (M1-F2); drop the
  CUDA/XPU platform gate; inherit the manager and block-count math (cache policy stays the
  upstream manager's — see M2-S1).
- Lazy factory registration in `spyre_inference/__init__.py`.

**Tests / runs**

- `test_spec_registration.py` (CPU-only): importing the plugin registers the spec; the factory
  resolves it.
- **M1 acceptance run** (`vllm serve … spec_name: SpyreOffloadingSpec`):
  - Server boots and reaches `register_kv_caches` without raising.
  - **Load/restore KVC from HBM:** a prefix-extending second prompt reports a host-tier hit
    (`N > 0` blocks loaded from host).
  - **Generation accuracy:** with `temperature=0`, tokens are **byte-identical** to a
    no-offload baseline (reloaded KV yields the exact same output).
- **Performance benchmark (recorded in the PR):** per-block offload (D2H) / reload (H2D)
  latency and throughput (GB/s), and TTFT reduction on a cache hit vs. full recompute.

**Acceptance**

- [ ] Registration resolves; no changes to the Spyre worker or platform (verified by diff).
- [ ] Boot + `register_kv_caches` OK; host-tier hit on the second prompt.
- [ ] `temperature=0` output byte-identical to baseline.
- [ ] Benchmark numbers recorded.

**Blocked by:** M1-S2

---

## M2-F1 — `SharedHostMetadata` (block-hash → slot directory)

**Title:** `[M2-F1] Hardware runtime: SharedHostMetadata (block-hash → slot directory)`
**Labels:** `kvc-offloading` · **Milestone:** M2 (end Sep 2026) · **Epic:** #\<M2>

**Body:**

**Summary:** A shared directory mapping block-hash → slot-id with per-slot lifecycle state and
a chunk descriptor for the multi-chunk (1p5) path. This is the bookkeeping that makes
cross-instance reuse possible.

**What**

- `create_or_attach(...)`; `lookup(hash) -> slot | miss`, `claim(hash) -> slot`,
  `publish(hash, slot)`, `evict(hash)`.
- Per-slot lifecycle: empty → reserved → valid → empty.
- Chunk descriptor `{num_chunks, [{domain_id, size}]}` stored per slot (consumed by M2-F3).
- Metadata segment mapped at a common virtual base (pointer-based internals); the data pool
  (M1-F2) stays index-addressed.

**Unit tests**

- claim/publish/lookup/evict across two attached processes behave correctly.
- Lifecycle transitions enforced (e.g. lookup before publish → miss; double-publish rejected).
- Chunk descriptor round-trips a multi-chunk allocation's shape.
- **Directory ↔ data-pool connection (required, single-process):** wire the metadata directory
  to the **M1-F2 shared data pool** and verify a full hash-addressed round-trip end-to-end:
  `claim(hash) -> slot_id` → `copyRaw` D2H a real device KV page into that slot →
  `publish(hash, slot_id)` → `lookup(hash)` returns the **same** slot_id → `copyRaw` H2D from
  that slot → assert **byte-for-byte** equal to the original page. **Single process is
  sufficient** — no concurrency control is introduced yet (that is M2-F2). This is the first
  test proving the directory (hash→slot) and the data pool (slot→bytes) compose into a working
  hash-addressed store.

**Acceptance**

- [ ] Directory ops correct across two processes.
- [ ] Lifecycle enforced; chunk descriptor round-trips.
- [ ] **Directory + M1-F2 data pool compose (single-process):** a KV page stored under a
      block-hash via `claim`/`publish` is retrieved byte-exact via `lookup` → `copyRaw` H2D
      (hash-addressed round-trip verified end-to-end).

**Blocked by:** M1-F2

---

## M2-F2 — Concurrency protocol (locks + generation + publish gate) — full race coverage

**Title:** `[M2-F2] Hardware runtime: concurrency protocol — full data-race coverage`
**Labels:** `kvc-offloading` · **Milestone:** M2 (end Sep 2026) · **Epic:** #\<M2>

**Body:**

**Summary:** The correctness protocol over the directory: a process-shared directory lock, a
per-slot read/write pin (read = reload, write = evict), a per-slot generation counter for
reuse/ABA detection, and a publish-on-DMA-completion gate. A stale or mid-write slot must
degrade to a **cache miss, never torn bytes**.

**What**

- Process-shared directory lock; per-slot rwlock-style pin; per-slot generation counter;
  publish-on-completion gate.

**Unit / stress tests — full data-race matrix (full coverage on the racing cases)**

Each case runs under concurrent multi-process load and asserts either correct data or a clean
miss — never a torn read:

- **Reader vs. evictor:** reader reloads a slot while another instance evicts + re-DMAs it →
  reader sees pre-evict bytes (pin held) or a miss, never a mixture.
- **ABA / generation reuse:** slot evicted and re-claimed for a *different* hash between a
  reader's `lookup` and its DMA → generation check fails → miss.
- **Publish-gate visibility:** reader `lookup`s a slot mid-write (claimed, DMA not complete,
  not yet published) → miss (mapping invisible until `publish`).
- **Concurrent claim of the same hash:** two instances `claim` the same absent hash at once →
  exactly one slot allocated, the other observes the winner (no double-allocation).
- **Concurrent claim under exhaustion:** claims race with the pool full → eviction picks a
  victim safely (victim's pin respected), no slot handed out twice.
- **Writer vs. writer:** two instances offload *different* hashes concurrently → no
  cross-contamination; both publish correctly.
- **Reader during re-DMA of the same hash:** hash re-offloaded while a reader holds a read pin
  → reader completes on the old generation or misses; writer waits or picks another slot.
- **Sustained stress:** run the matrix for a sustained duration under N processes → **zero
  torn reads, zero double-allocations**.

**Acceptance**

- [ ] All matrix cases pass.
- [ ] Sustained N-process stress reports zero torn reads and zero double-allocations.

**Blocked by:** M2-F1

---

## M2-F3 — `copyRaw` multi-chunk (1p5) + cross-process slot round-trip

**Title:** `[M2-F3] Hardware runtime: copyRaw multi-chunk (1p5) + cross-process slot round-trip`
**Labels:** `kvc-offloading` · **Milestone:** M2 (end Sep 2026) · **Epic:** #\<M2>

**Body:**

**Summary:** Extend the raw copy for the multi-chunk (1p5) case. The `slot_id` seam and copy
signature are unchanged from M1; single-chunk is the degenerate `num_chunks == 1` case.

**What**

- **Store (offload):** walk the source `CompositeAddress`, pack chunks contiguously into the
  slot, record the `{num_chunks, [{domain_id, size}]}` descriptor into the directory (M2-F1).
- **Load (reload):** read the descriptor, place chunks on their recorded domains to build a
  **fresh** `CompositeAddress`, DMA each chunk from its sub-offset in the slot.

**Unit tests**

- A **multi-chunk** device page stored into a slot from one process reloads **byte-for-byte**
  from another process into a fresh same-`(shape, dtype)` page.
- Copy size owned by the hardware runtime (`total_size()`).
- Single-chunk continues to round-trip byte-exact (regression).

**Acceptance**

- [ ] Cross-process multi-chunk slot round-trip is byte-exact.
- [ ] Single-chunk still byte-exact; size owned by the hardware runtime.

**Blocked by:** M1-F1, M1-F2, M2-F2

---

## M2-T1 — `SharedHostMetadata` (+ shared-pool attach) pybind passthroughs

**Title:** `[M2-T1] torch-spyre: SharedHostMetadata (+ shared-pool attach) pybind passthroughs`
**Labels:** `kvc-offloading` · **Milestone:** M2 (end Sep 2026) · **Epic:** #\<M2>

**Body:**

**Summary:** Add the pybind passthrough for the hardware runtime's `SharedHostMetadata`
directory and the shared-attach usage of `SharedHostPool`. torch-spyre adds nothing — no
shared-segment creation, no locking, no directory logic.

**What**

One-to-one pybind exposure of the hardware runtime's `SharedHostMetadata`, plus the
shared-attach usage of `SharedHostPool` (the same object as M1-F2). torch-spyre adds no
logic — these are thin passthroughs to the hardware-runtime header.

**Exact Python calls to expose (integer slot-id is the only seam):**

- `SharedHostMetadata`:
  - `SharedHostMetadata.create_or_attach(stream, name, capacity) -> SharedHostMetadata`
    — create or attach the shared directory segment by `name`.
  - `md.lookup(block_hash: int) -> int` — return the slot-id for a published hash, or a
    sentinel **miss** (e.g. `-1`) if absent. (No pointer returned — a slot-id.)
  - `md.claim(block_hash: int) -> int` — reserve and return a slot-id for a hash (empty →
    reserved).
  - `md.publish(block_hash: int, slot_id: int) -> None` — mark a claimed slot valid/visible.
  - `md.evict(block_hash: int) -> None` — release a slot (→ empty).
  - `md.capacity() -> int` (and any count/occupancy accessor the header defines).
- `SharedHostPool` (shared-attach usage; the object itself is bound in M1-T2):
  - `SharedHostPool.create_or_attach(stream, name, num_slots, slot_bytes) -> SharedHostPool`
    used with a **shared name** so a second process attaches the same slots.
- **Not exposed:** `slot_ptr(i)` and any raw host/device address — these stay inside the
  hardware runtime. The Python seam is exclusively the integer `slot_id` / `block_hash`.

The exact names/arities are pinned to the hardware-runtime `SharedHostMetadata` header at
implementation time; the list above is the binding contract this issue must satisfy (adjust
only to match the merged header, and note any deviation in the PR).

**Unit tests**

- Each exposed call is reachable from Python with the signature above: `create_or_attach`,
  `lookup`, `claim`, `publish`, `evict`, `capacity`.
- `lookup` on an unpublished hash returns the **miss sentinel**; after `claim` + `publish`,
  `lookup(block_hash)` returns that slot-id; after `evict`, `lookup` misses again.
- Create-or-attach the pool + directory from two processes → observe the **same slots and the
  same block-hash → slot mappings** (driven from Python).
- **Concurrency-safe ops pass through faithfully (from Python):** the bindings expose the
  protocol's guarantees without weakening them — a slot that is mid-write / not-yet-published
  (or reused for a different hash) yields a **miss from Python**, never torn bytes. This is the
  right layer for the "torn-read visible from Python" assertion (needs only the bindings; the
  full race matrix itself is proven in M2-F2, and the connector-level miss→recompute behavior is
  M2-S2).
- No raw host pointer or device address crosses into Python (only ints).

**Acceptance**

- [ ] All listed calls are exposed with the stated signatures and integer-only seam.
- [ ] `lookup`/`claim`/`publish`/`evict` semantics verified from Python (miss → claim →
      publish → hit → evict → miss).
- [ ] Two-process attach sees the same slots and mappings from Python.
- [ ] **Torn-read passthrough:** a mid-write / reused slot yields a **miss from Python**, never
      torn bytes (bindings do not weaken the M2-F2 protocol).
- [ ] No pointer/address leaks to Python.

**Blocked by:** M2-F1, M2-F2 — the concurrency-safe ops must exist before their Python
passthrough (incl. the torn-read miss) can be tested.

---

## M2-S1 — `SpyreSharedOffloadingSpec` + registration

**Title:** `[M2-S1] spyre-inference: SpyreSharedOffloadingSpec + registration`
**Labels:** `kvc-offloading` · **Milestone:** M2 (end Sep 2026) · **Epic:** #\<M2>

**Body:**

**Summary:** Implement `shared_spec.py` — `SpyreSharedOffloadingSpec` subclassing M1's
`SpyreOffloadingSpec`, reusing its handlers and copier **unchanged** (same canonical
`copy_tensor_raw(dev_tensor, pool, slot_id, to_device)`). The only difference from M1 is the
host destination: a slot in a **shared** pool named by integer `slot_id` via a **shared**
`SharedHostMetadata` directory, instead of a private pool.

**Who owns the cache policy (answers a recurring question)**

The **cache policy is owned upstream, not by us.** `SpyreOffloadingSpec` (M1-S3) subclasses the
upstream vLLM CPU `OffloadingSpec`, which brings the upstream `OffloadingManager` — that manager
owns **admission, eviction (LRU), block-hash bookkeeping, and the hit/miss decision**. Our Spyre
specs override only the **transfer mechanism** (which handlers/copier move the bytes), never the
policy. So `SpyreSharedOffloadingSpec` **inherits the same upstream cache policy** transitively
(upstream `OffloadingSpec` → `SpyreOffloadingSpec` → `SpyreSharedOffloadingSpec`).

What M2 changes is **not** the policy but **where a decided hit resolves to storage**: in M1 the
manager's slot bookkeeping is process-local; in M2 the block-hash → slot mapping is externalized
to the **shared** `SharedHostMetadata` directory so a hit can resolve to a slot another instance
published. Eviction still runs upstream; when the manager evicts a block we call
`SharedHostMetadata.evict(hash)` to release the shared slot. If we ever need Spyre-specific
admission/eviction behavior, that is a **separate, explicitly-scoped** manager override — out of
scope for M1/M2, which deliberately reuse the upstream policy verbatim.

**What**

- `shared_spec.py`: attach a shared `SharedHostPool` + `SharedHostMetadata`; on store `claim`
  a slot → D2H raw copy → `publish` after sync; on load `lookup` → H2D raw copy.
- Third lazy factory registration in `spyre_inference/__init__.py`; inert when not selected
  and on builds without the M2 hardware-runtime surface.

**Tests**

- Spec resolves via the factory.
- Importing the plugin on a build **without** the M2 surface does not error (lazy import).
- Diff shows the copier/handler code is byte-for-byte the M1 code (only pool
  construction/sharing changes).

**Acceptance**

- [ ] Spec resolves; import inert without the M2 surface; M1 path unaffected.
- [ ] Copier/handler reused unchanged (verified by diff).
- [ ] **Cache policy not overridden:** admission/eviction/hit-miss remain the upstream
      `OffloadingManager`'s (verified by diff — no policy override); M2 changes only pool
      construction/sharing and the shared directory that backs slot resolution.

**Blocked by:** M2-T1, M1-T3, M2-F3, M1-S3

---

## M2-S2 — Shared-pool round-trip + connector miss→recompute test

**Title:** `[M2-S2] spyre-inference: shared-pool round-trip + torn-read test`
**Labels:** `kvc-offloading` · **Milestone:** M2 (end Sep 2026) · **Epic:** #\<M2>

**Body:**

**Summary:** A **connector-level** shared-pool round-trip test, plus a test that the connector
handles a **miss correctly** (falls back to recompute) when the lower layers report one. This
issue does **not** re-prove concurrency correctness — that is owned and exhaustively tested where
the protocol lives (see below).

**Where the race-correctness proof lives (answers "why is this in spyre-inference?")**

The "miss, never torn bytes" **guarantee** is a property of the hardware-runtime concurrency
protocol, so it is **proven at the layer that owns it, not re-proven here**:

- **M2-F2 (hardware runtime)** owns and exhaustively tests the full data-race matrix
  (reader-vs-evictor, ABA/generation reuse, publish-gate, concurrent-claim, sustained N-process
  stress). That is where torn-read freedom is *established*.
- **M2-T1 (torch-spyre)** proves the **Python bindings faithfully pass the protocol through**:
  once the data + metadata pool bindings and their concurrency-safe ops are exposed, a mid-write
  `lookup` observed **from Python** returns a miss (binding does not bypass or weaken the pin /
  generation / publish-gate). This is the right place for the "torn-read visible from Python"
  assertion — it needs only the bindings, not the connector.
- **M2-S2 (here, spyre-inference)** verifies only the **connector's** behavior *given* those
  guarantees: a shared-slot round-trip succeeds, and when the layers below return a **miss**, the
  connector degrades to **recompute** (no crash, no corrupt output) rather than serving stale
  bytes. It does not construct races itself.

**Tests**

- `test_shared_pool_round_trip.py` (Spyre-gated): store a known-pattern device page into a slot
  (`claim` + D2H raw copy + `publish`) → `lookup` + H2D raw copy into a fresh page → assert
  **byte-exact** content.
- `test_connector_miss_recompute.py`: with `lookup` forced to report a miss (stubbed/injected at
  the binding boundary — **not** by racing the pool here), the connector takes the recompute path
  and produces correct output; no torn/stale bytes are ever consumed.

**Acceptance**

- [ ] Byte-exact round-trip through a shared slot (connector level).
- [ ] On a reported miss, the connector **recomputes** cleanly (correct output, no stale/torn
      bytes consumed). The **race-correctness proof itself lives in M2-F2 (protocol) and M2-T1
      (binding passthrough)** — not duplicated here.

**Blocked by:** M2-S1

---

## M2-S3 — Cross-instance test + M2 acceptance & benchmark

**Title:** `[M2-S3] spyre-inference: cross-instance test + M2 acceptance & benchmark`
**Labels:** `kvc-offloading` · **Milestone:** M2 (end Sep 2026) · **Epic:** #\<M2>

**Body:**

**Summary:** Two-process cross-instance test plus the M2 end-to-end acceptance and a
cross-instance benchmark.

**Tests / runs**

- `test_cross_instance.py` (two-process): process A stores+publishes a block into the shared
  pool; process B attaches the same named pool + directory, `lookup`s the same content hash,
  reloads it → assert a cross-instance hit and byte-identical reload.
- **M2 acceptance run** (two `vllm serve` instances on one host, same shared pool):
  - **Cross-instance peer hit:** B gets a host-tier hit **on its first request** on a block A
    offloaded — device←host DMA, no recompute, no disk.
  - **Generation accuracy:** with `temperature=0`, B's tokens are **byte-identical** to a
    no-cache baseline.
- **Performance benchmark (recorded in the PR):** cross-instance peer-hit latency (B's TTFT on
  a shared-pool hit) vs. full recompute on B, and shared-pool reload throughput vs. M1
  (single-instance) numbers.

**Acceptance**

- [ ] Cross-instance peer hit works; byte-identical reload.
- [ ] `temperature=0` output byte-identical to baseline.
- [ ] Benchmark numbers recorded.
- [ ] M1 path unaffected (`pytest … kv_offload` green).

**Blocked by:** M2-S2

---

## M2-T2 — `get_dma_stream` accessor (DEFERRED backlog — no milestone)

**Title:** `[M2-T2] torch-spyre: get_dma_stream accessor (DEFERRED backlog — no milestone)`
**Labels:** `kvc-offloading` · **Milestone:** none (deferred backlog) · **Epic:** #\<M2> (deferred)

**Body:**

**Summary:** *(Deferred / backlog — an optimization, **not** part of M1 or M2.)* A thin wrapper
over the pooled-stream accessor so the connector can keep a **dedicated** DMA stream for
offload/reload, overlapping the compute stream. Per review (Yue): M2 focuses on the correctness
**baseline** (shared pool + directory + concurrency); a dedicated DMA stream is a throughput
optimization layered on top of a working baseline, so it is pulled out of M2 and tracked here for
later scheduling. It gates nothing: both M1 and M2 copy on the current/default stream.

**What**

- `get_dma_stream(device=None) -> SpyreStreamHandle` over the existing pooled-stream accessor.
- Copy bindings accept an optional explicit stream; when omitted they use the current stream for
  the device (single-stream fallback — the M1/M2 baseline behavior).

**Unit tests**

- Returns a usable stream handle for a valid device; copies issued on it complete correctly.
- With `get_dma_stream` absent/unused, the connector still works on the default stream
  (single-stream fallback verified).

**Acceptance**

- [ ] Dedicated stream works; single-stream fallback works.
- [ ] Measured overlap benefit recorded vs. the single-stream baseline (this is the
      optimization's justification).

**Milestone:** none — **deferred backlog**. Schedule only after the M2 baseline lands.

**Blocked by:** the torch-spyre multi-stream support landing (external), and the M2 baseline
(shared pool + directory + concurrency) being in place so overlap can be measured against it.
