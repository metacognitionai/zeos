# ZEOS -- Implementation

**Audience:** engineers who will extend, port, or productionise this codebase.

The rest of this corpus is design. This document is the implementation dimension:
what is real versus simulated, where the code lives, and the backend interface a
real serving stack must implement. Nothing conceptual is defined here.

---

## 1. What is real and what is simulated

This distinction matters more than anything else in this document, and it is
maintained in the type system rather than in comments.

| Aspect | Status |
| --- | --- |
| Scheduling, preemption, suspension stack, resume | **Real** |
| Pipes, backpressure, blocking, vectors, storm control | **Real** |
| Segment table, rings, provenance, capability checks | **Real** |
| Block structure, alignment, padding | **Simulated but faithful** |
| Control-token unforgeability | **Structural** -- a `CONTROL` token cannot be decoded unless the kernel enables it |
| Eviction, stubs, page faults, store round trip | **Real mechanism** |
| **Attention mass** | **Synthetic** -- the scripted backend cannot measure; the kernel resolves hints |

Nothing about eviction regret, θ-parameter sensitivity, or taint-creep rates can be
concluded from any run of this code. The mechanisms are validated; the policies are
not, and cannot be until a real serving stack supplies measured attention.

---

---

## 2. Module map

```
src/zeos/
├── core/                    THE KERNEL -- stdlib-only, no I/O, no wall-clock
│   ├── ids.py               identifiers + cross-cutting enums (leaf: imports nothing)
│   ├── serde.py             structural round-trip, hard failure on unserialisable
│   ├── clock.py             two time bases, both injected
│   ├── events.py            the 47-event journal alphabet
│   ├── pcb.py               Job -- descriptor + transcript + metadata
│   ├── scheduler.py         ready set, running job, suspension stack, inheritance
│   ├── pipes.py             bounded buffers, blocking, select, backpressure
│   ├── vectors.py           vector table, coalesce/queue/reentrant, throttling
│   ├── segments.py          segment table, block alignment, masking          [MP]
│   ├── integrity.py         Biba low-water-mark, θ_read                      [MP]
│   ├── capabilities.py      capability table, schemas, rate limits           [MP]
│   ├── store.py             content-addressed span store                     [VM]
│   ├── residency.py         eviction planner, stubs, thrash, admission       [VM]
│   ├── pager.py             fault service, NEED routing, page-in placement   [VM]
│   ├── faults.py            fault taxonomy, policy resolution, notice text
│   └── kernel.py            the state machine -- tick(), ~1400 lines
├── machine/
│   ├── base.py              THE BACKEND SWAP POINT -- five ops + serving contract
│   └── scripted.py          scripted streams, blocks, synthetic attention
├── descriptor/
│   ├── schema.py            frontmatter → Descriptor, strict parsing
│   ├── loader.py            markdown + YAML, case directories
│   └── lint.py              load-time rejection: core + MP + VM rules
├── nli/                     instruction as compilation -- deliberately the untrusted half
│   ├── envelope.py          utterance envelopes, deixis, the local safety vocabulary
│   ├── compiler.py          utterance → invocation / mission / synthesis artifact
│   └── dispatcher.py        compile, gate, echo back, dispatch
├── journal/                 codec (byte-stable) + writer (streaming)
├── transport/               base.py (the seam), local.py, link.py (injected latency/loss)
├── world/store.py           namespaced state, read/write sets, resume diffs
├── monitor/                 the journal folded into a picture of the system (`top`)
├── debugger/                the same fold, drawn: wiring, scrubber, token stream
│   ├── payload.py           case + journal -> one JSON object; pure, tested
│   ├── server.py            page assembly; self-contained export, or a local server
│   └── static/              the page itself -- no CDN, no vendored library
├── demo/                    problem/solution harness; no case ships here, see README
│   ├── problem.py           Contract, Problem, conformance validation
│   ├── solution.py          descriptor tree only
│   ├── criteria.py          11 criterion kinds, journal-evaluated
│   ├── runner.py            bind → run → score
│   └── harness.py           run_behaviour -- the unit tier, reusable
├── driver.py                ALL I/O and time, outside the core
├── federated.py             two kernels, one process: partition, restore, replicate
└── cli.py                   lint / run / replay / inspect / debug / demo
```

### Dependency rule

`core/` imports only the standard library plus `core/`, `world/`,
`descriptor/schema.py`, `machine/base.py`, and `nli/` -- each of which is itself
stdlib-only. It never imports `descriptor/loader.py` (which uses PyYAML),
`driver.py`, or `demo/`. `core/ids.py` is a leaf that imports nothing from `zeos`,
which is what keeps the package cycle-free. The whole rule is enforced by
`tests/contract/test_core_is_stdlib_only.py`, which checks both the declared
imports and the transitive import closure.

---

---

## 3. The machine backend interface -- where a real backend plugs in

`machine/base.py` is the most important interface in the system. Everything above it
is written against this and nothing else.

The contract deliberately mirrors the four asks ZEOS makes of a serving stack, so a
backend satisfying this interface satisfies the design.

### 3.1 The five ops

```python
def decode(self, job: JobId, *, allow_control: bool) -> DecodeResult
def inject(self, job: JobId, tokens: Sequence[Token]) -> tuple[int, int]
def trunc(self, job: JobId, at: int) -> int
def fork(self, parent: JobId, child: JobId) -> int
def splice(self, job, start, end, tokens) -> SpliceResult
```

### 3.2 The serving-stack contract

```python
def set_mask(self, job: JobId, allowed_blocks: frozenset[int]) -> None
def visible_blocks(self, job: JobId) -> frozenset[int]
def pad_to_block(self, job: JobId) -> int
def blocks_for_range(self, job, start, end) -> frozenset[int]
def transcript(self, job: JobId) -> tuple[Token, ...]
```

Four asks, all of which exist in current stacks in some form and none of which
requires model changes:

1. allowed-block bitmap fed to attention (`set_mask`)
2. per-segment attention mass aggregated per block (`DecodeResult.attention`)
3. reserved-token control in tokenizer and sampler (`allow_control`)
4. grammar-constrained decoding for endorser schemas (`Capability.schema`)

### 3.3 Division of responsibility -- easy to get wrong

- The **machine** owns the materialised token sequence and its blocks. It knows
  offsets and lengths, and it enforces the mask.
- The **kernel** owns the segment table, rings, integrity, and residency. Segment
  metadata is PCB state, **not context**.

They meet at token offsets. The machine reports what it did; the kernel decides what
it means. Getting this backwards would put protection metadata inside the window
where the model could read and forge it.

### 3.4 The attention split -- where the fiction lives

```python
@dataclass(frozen=True, slots=True)
class DecodeResult:
    tokens: tuple[Token, ...]
    request: MachineRequest
    attention: Mapping[int, float] | None      # MEASURED, per KV block
    attention_hint: AttentionHint | None       # SYNTHETIC, resolved by the kernel
    at_block_boundary: bool
```

A real backend populates `attention` and leaves `attention_hint` None.
`ScriptedMachine` does the reverse. Splitting them in the type system is slightly
ugly, and the ugliness is the point: it marks exactly where the fiction lives so no
policy claim can rest on it by accident.

`attention` is **block-granular** because that is what a paged serving stack actually
aggregates. Since segments are block-aligned by construction, the kernel sums per
segment *exactly* rather than approximately.

### 3.5 Implementing a vLLM backend

Order of work, roughly:

1. `create_context` / `destroy_context` → sequence lifecycle.
2. `decode` → one step, returning tokens and any reserved-token request.
3. `inject` → prefill of foreign tokens at the tail; return the offset range.
4. `set_mask` / `visible_blocks` → the allowed-block bitmap into the attention
   kernel. Target overhead **< 2% decode throughput**.
5. `attention` → per-block mass aggregation. One reduction per block; sampling every
   k-th block is an acceptable degradation.
6. `splice` → the expensive one. Report `invalidated_downstream` honestly; the
   eviction planner's cost model consumes it.
7. `fork` → copy-on-write over KV blocks.

`allow_control` maps onto sampler-side reservation. `Capability.schema` maps onto
grammar-constrained decoding.

---
