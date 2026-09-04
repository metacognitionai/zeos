# ZEOS-AM: The Abstract Machine

**Status:** specification v0.1 (2026-08-19). Companion to ZEOS-MP (protected mode). Normative for `zeos.machine`.

**A note on the name.** In this corpus **VM already means Virtual Context**, the demand-paging layer. The thing this document specifies is the abstract machine that ZEOS executes on: the five-operation interface in `src/zeos/machine/base.py` that the kernel drives and that a real serving stack must implement. Calling it "the VM" would collide with a document that already exists and means something else, so it is **AM**, the abstract machine. Where this document says "the machine" it means AM.

# 1. Why this document exists

`MachineBackend` is the interface the kernel calls to make a transformer do work. Its own docstring calls it "the most important interface in the system, because it is the one M1 swaps for a real paged-KV serving stack." Everything else in ZEOS is portable across backends; this is the seam.

The interface is currently defined by a Protocol with good prose comments and one implementation, `ScriptedMachine` (M0), pinned by 18 unit tests. That is enough to build a kernel against. It is **not** enough to hand to someone building M1, because the comments describe intent while the tests pin only the cases someone thought to write. The gap between those two is where an M1 backend will silently diverge: it will satisfy every test, honour every docstring, and still break the kernel, because the obligations it violated were never written down.

This specification closes that gap. It states what the machine's state **is**, which invariants hold over it, what each operation guarantees, and which of those guarantees an M1 implementer must reproduce versus which are M0 conveniences. It is deliberately **retrospective**: it formalises what the implementation already does, verified by probing it, rather than proposing something new. Where the implementation and the intent disagree, this document records the disagreement instead of quietly picking a side.

Three classes of statement appear, and every normative clause is tagged:

| Tag | Meaning |
| --- | --- |
| **MUST** | Required of every backend. The kernel's correctness depends on it |
| **POLICY** | Deliberately left to the backend. The kernel must not assume either way |
| **M0** | True of `ScriptedMachine` only. An M1 backend is not required to reproduce it |

# 2. Position in the system

The machine and the kernel divide one piece of state between them, and the division is the reason the interface is small.

```
        kernel                                machine
   ┌──────────────────────┐            ┌──────────────────────┐
   │ segment table        │            │ token sequence T     │
   │ rings, integrity     │            │ blocks (derived)     │
   │ residency, working   │            │ the mask M           │
   │   set, capabilities  │            │ decode state         │
   │ scheduling           │            │                      │
   └──────────┬───────────┘            └──────────┬───────────┘
              │                                   │
              └───────── token offsets ───────────┘
                    the entire shared vocabulary
```

**The machine owns the materialised token sequence and its block structure. The kernel owns the segment table, rings, integrity and residency. They meet at token offsets, and nowhere else.**

Consequences that matter:

- The machine knows nothing of rings, integrity, capabilities, priorities or jobs-as-processes. It takes a `JobId` purely as a context handle. A backend author needs no ZEOS security model.
- Every protection decision is the kernel's. The machine's only enforcement duty is the mask (§8) and control-token gating (§9), and both are mechanical.
- **Offsets are the whole contract.** An operation that moves an existing token to a different offset invalidates the kernel's segment table, so exactly one operation is permitted to do it and it must report enough for the kernel to repair itself (§6.5).

# 3. State

## 3.1 Tokens

```
Token = (text: str, kind: TokenKind)
TokenKind = NORMAL | CONTROL
```

A token is an opaque unit of context. **MUST:** the machine may not interpret `text`. **MUST:** `kind` partitions the vocabulary into what a model may emit and what only the kernel may introduce (§9).

**M0:** tokens carry text and are produced by whitespace splitting (`tokens_from_text`), so one "token" is one word. This is a stand-in for a tokenizer. **POLICY:** the mapping from strings to tokens. No kernel logic may assume one token is one word, and none currently does; the kernel counts tokens and never inspects text except when rendering a transcript for output. It does **record** it: the token-bearing events (`machine.decode`, `machine.inject`, `pipe.written`, `pipe.read`) carry the text alongside the count, so what a job generated and what entered its context are answerable from the journal rather than only from a live machine. Recording is not inspecting -- no kernel decision reads these fields -- and the text is a tuple of one string per token rather than a rendered line, because a real tokenizer's tokens are sub-word pieces.

## 3.2 Context

The machine holds a partial map from job identity to context:

```
ctx : JobId ⇀ Context
Context = (T, M, D)
```

- `T` is the **materialised token sequence**, a finite sequence indexed from 0. Write `|T|` for its length and `T[i]` for its elements.
- `M` is the **mask**: either `⊥` (no mask set) or a set of block indices. `⊥` and `∅` are **distinct** (§8.2).
- `D` is **backend-private decode state**. The kernel may not observe or depend on it. **M0:** `D` is a script plus a program counter.

`B`, the **block size**, is a positive integer fixed for the machine's lifetime and exposed as `block_size`. **MUST:** `B` is constant. The kernel reads it once when constructing a job's segment table (`SegmentTable(self.machine.block_size)`), so a backend that changed it would desynchronise every table already built.

## 3.3 Derived quantities

None of these are stored. All are functions of `T` and `M`, and **MUST** be computed as such:

```
resident_tokens(j)     = |T|
blocks(j)              = ⌈|T| / B⌉                    (so blocks = 0 when T is empty)
block_of(i)            = ⌊i / B⌋
blocks_for_range(s, e) = { block_of(i) : s ≤ i < e }  (empty when s ≥ e)
visible(j)             = { 0 .. blocks(j)-1 }                 if M = ⊥
                       = { 0 .. blocks(j)-1 } ∩ M              otherwise
```

`ContextStats` reports the first two plus `open_segment_tokens = |T| mod B`.

> **Naming defect.** `open_segment_tokens` does not report the open *segment*, which is a kernel concept the machine cannot see. It reports tokens past the last block boundary. The kernel's own open segment may be longer or shorter. The field is correctly used today but the name invites exactly one wrong inference, and an M1 author should read it as `tokens_past_last_boundary`.

# 4. Invariants

These hold at every point at which the kernel may observe the machine, that is between operations.

**AM-I1 (density).** `T` is contiguous. Offsets `0` to `|T|-1` are all occupied and there are no holes. A splice that shortens a span closes the gap rather than leaving one. This is why `SegmentTable.evict_to_stub` renumbers everything downstream, and why it must.

**AM-I2 (blocks are derived).** Block structure is a function of `|T|` alone. Blocks are never stored, never sparse, and never reordered. Block `k` holds offsets `kB` to `min((k+1)B, |T|) - 1`. Only the final block may be partial.

**AM-I3 (mask enforcement is hard).** If `M ≠ ⊥`, the backend **MUST NOT** allow content in a block outside `visible(j)` to influence decoding, and **MUST NOT** report attention mass against such a block. This is enforcement, not a request the model may decline. A violation is a backend bug and is signalled by `MaskViolation`.

**AM-I4 (control tokens are unforgeable).** `decode` **MUST NOT** return a token of kind `CONTROL` unless called with `allow_control=True`. A violation raises `ControlTokenViolation`. Note the asymmetry in §9: `inject` is unrestricted.

**AM-I5 (offset stability).** Of the five operations, **only SPLICE may change the offset of a token that already exists**. `INJECT` appends. `TRUNC` removes a suffix. `DECODE` appends. `FORK` writes a different context. Any backend that relocates existing tokens for any other reason, including compaction or defragmentation, breaks the kernel's segment table.

**AM-I6 (determinism).** Every operation is a function of the observable context, its arguments, and backend-private state that is itself a deterministic function of the operation history. No wall clock, no unseeded randomness, no dependence on batching, timing, or concurrency -- the determinism gate (byte-identical journals) rests on this.

**AM-I7 (fail closed).** Where the mask and the token sequence disagree, the intersection wins and visibility shrinks. A stale mask can only ever hide content, never reveal it (§8.3).

# 5. Operation summary

| Operation | Mutates `T` | Moves existing offsets | Errors |
| --- | --- | --- | --- |
| `decode` | appends | no | `ScriptExhausted` (M0), `ControlTokenViolation` |
| `inject` | appends | no | none |
| `trunc` | truncates | no | `IndexError` |
| `fork` | writes child | no (child only) | `KeyError` on unknown parent |
| `splice` | replaces a span | **yes** | `IndexError` |
| `pad_to_block` | appends | no | none |
| `set_mask` | no | no | none |
| `create_context` / `destroy_context` | creates / drops | no | none |
| `stats`, `visible_blocks`, `blocks_for_range`, `transcript` | no | no | `KeyError` if no context |

**MUST:** every operation taking a `JobId` raises if no context exists for it. **M0** raises `KeyError` with the text `no context for job N; create_context first`. **POLICY:** the exception type.

# 6. The five operations

## 6.1 DECODE

```
decode(job, *, allow_control: bool) -> DecodeResult
DecodeResult = (tokens, request, attention, attention_hint, at_block_boundary)
```

One decode step is **one scheduling quantum**. This is what makes "preempts within one token boundary" a checkable property rather than a slogan: the kernel regains control after every decode and no job can run longer than one step without being rescheduled.

**Precondition:** a context exists for `job`.

**Postcondition:** `T' = T · tokens`. The returned `tokens` are appended in order. `|tokens|` may be zero.

**MUST:** append only. A decode may not modify, reorder or remove existing tokens.

**MUST:** if `allow_control` is false and any returned token has kind `CONTROL`, raise `ControlTokenViolation` rather than returning it. **M0** raises after advancing its program counter, so the step is consumed; this is unobservable to a correct kernel because the violation is fatal.

**`request`** is the job's request to the kernel, a `MachineRequest` (§10). `OpKind.NONE` means the step produced text and asked for nothing.

**`attention` and `attention_hint`** are the two attention channels (§7).

**`at_block_boundary`** is **advisory and currently unused**. See §6.1.1; it does not mean what an implementer would assume.

**Errors. M0:** decoding past the end of a script raises `ScriptExhausted`, which is a fixture authoring error rather than a runtime condition. **MUST:** a real backend needs a defined answer for end-of-generation. That answer is an open question (OQ-AM-1); today the kernel relies on scripts ending in `exit`, which is an `OpKind.EXIT` request, so termination is expressed *in band* as a request rather than out of band as an exhaustion. **An M1 backend should map EOS to an `EXIT` request, not to an exception.**

### 6.1.1 The boundary signal does not mean "crossed a boundary"

`at_block_boundary` is computed as:

```
after % B == 0 and after != before
```

That is **"landed exactly on a boundary"**, which is not the same as **"crossed into a new block"**. With `B = 4`, a decode taking `|T|` from 3 to 6 crosses the boundary at 4 and reports `False`. Verified:

```
B=4, start len=3
  decode +3 -> len=6  blocks=2  at_block_boundary=False   <- crossed, not signalled
  decode +2 -> len=8  blocks=2  at_block_boundary=True
  decode +2 -> len=10 blocks=3  at_block_boundary=False   <- crossed, not signalled
```

**The kernel does not use this field.** It derives boundaries itself, from the token count:

```python
def _current_block(self, job):
    return self.machine.stats(job.job_id).resident_tokens // self.machine.block_size
```

and fires `_maybe_boundary` when that value *increases*. That predicate is correct under multi-token steps, which is why watermark demotion, mask changes and eviction batching all work today despite the flag being wrong for the purpose it appears to serve.

**MUST:** boundary work is the kernel's responsibility and is derived from `stats()`. A backend **MUST NOT** assume the kernel consumes `at_block_boundary`, and the kernel **MUST NOT** start consuming it without redefining it. Recommended resolution: delete the field, or redefine it as `blocks_crossed: int`. Recorded as OQ-AM-2.

## 6.2 INJECT

```
inject(job, tokens) -> (start, end)
```

**Postcondition:** `T' = T · tokens`; returns `(|T|, |T| + |tokens|)`, a half-open range naming exactly where the tokens landed.

**MUST:** append only, and **MUST** return the range, because the kernel turns it directly into a segment record. This is the kernel's sole means of introducing content, so it carries the whole of total provenance: everything in a context arrived either by injection, from a known pipe with a known ring, or by decoding, from the model itself.

**MUST:** `inject` accepts `CONTROL` tokens without restriction. This asymmetry against `decode` is the security property, not an oversight (§9).

## 6.3 TRUNC

```
trunc(job, at) -> dropped
```

**Precondition:** `0 ≤ at ≤ |T|`, else `IndexError`. Note `at = |T|` is legal and is a no-op returning 0.

**Postcondition:** `T' = T[0:at]`; returns `|T| - at`.

**MUST:** offsets below `at` are unchanged. Offsets at or above `at` cease to exist.

**Mask interaction:** the mask is **not** adjusted, so it may now name blocks that no longer exist. By AM-I7 those simply drop out of `visible(j)`. Verified: with `B=4`, `|T|=12`, `M={2}`, a `trunc(4)` leaves `visible = ∅`, which denies all attention rather than granting it. Fail closed, but silent, so **the kernel MUST re-assert the mask after any operation that shrinks `T`** if it intends the job to keep running.

## 6.4 FORK

```
fork(parent, child) -> tokens_copied
```

**Postcondition:** the child's `T` becomes a copy of the parent's, and the child's `M` becomes the parent's `M`.

**Mask inheritance is deliberate and load bearing.** A compartment child is created by forking the parent and then *narrowing*, so it must start from the parent's visibility and lose ground. Verified: after `set_mask(parent, {0})`, a fresh child reports `visible = {0}`.

**Decode state is preserved, not inherited, when the child already has a context.** If a context exists for `child`, only tokens and mask are copied and the child keeps its own script and program counter. This matters for compartments: a compartment child runs *its own* behaviour over a slice of the parent's context, so inheriting the parent's decode state would be exactly wrong. If no context exists, **M0** copies the parent's script and counter as well.

**POLICY:** whether the copy is physical or copy-on-write. **M0** copies the token list, because at this scale copy-on-write is an accounting story rather than a memory one; a real backend shares KV blocks and the kernel's cost model already treats fork as cheap.

## 6.5 SPLICE

```
splice(job, start, end, tokens) -> SpliceResult
SpliceResult = (tokens_in, invalidated_downstream)
```

**This is the only operation that moves existing offsets, and the only one with a serious obligation on the caller.**

**Precondition:** `0 ≤ start ≤ end ≤ |T|`, else `IndexError`.

**Postcondition:** `T' = T[0:start] · tokens · T[end:]`. Returns `tokens_in = |tokens|` and `invalidated_downstream = |T| - end`, the count of tokens after the replaced span.

**Offset effect:** every offset at or above `end` shifts by `delta = |tokens| - (end - start)`. Verified: splicing `[2,4)` with one token over an 8-token context yields 7 tokens with everything from offset 2 onward shifted by `-1`, and reports `tokens_in=1, invalidated_downstream=4`.

**MUST (caller).** The kernel **MUST** renumber its segment table by `delta` for every segment after the spliced span. It does: `SegmentTable.evict_to_stub` computes `delta = stub_tokens - record.tokens` and adds it to the `start` and `end` of every following record. There is no other splice call site. **A second splice call site added without renumbering would silently corrupt every segment offset downstream**, and nothing in the type system prevents it, so this is the sharpest correctness hazard in the interface.

**Why `invalidated_downstream` exists.** On real hardware, a splice invalidates the KV cache for everything after `start`, because those positions' keys and values were computed against a prefix that has changed. The count is the recompute cost, and the eviction cost model is built on it. **MUST:** report it accurately. **POLICY:** whether the backend actually recomputes eagerly, lazily, or by position-encoding tricks that avoid it.

**Why splice exists at all,** given that it is the dangerous operation: paging a span out to a stub, and back in, must happen *in place*. Appending the paged-in content instead would leave the stub in the transcript ahead of the content it stands for and reorder the job's history. The cost is real and the kernel prefers spans with little after them precisely because of it.

# 7. Attention accounting

Attention is the machine's report of what the model actually used. It feeds two kernel mechanisms: integrity demotion (mass at or above `θ_read` counts as *used* rather than merely present) and eviction (the attention clock). It is reported through **two channels, and a backend supplies exactly one.**

## 7.1 The measured channel

```
attention: Mapping[int, float] | None      # block index -> mass
```

**MUST (M1):** a backend that can measure reports per **block**, and the kernel aggregates per segment by summing over `blocks_for(record)`. Segments are block-aligned by construction, so the sum is exact rather than approximate. This is the intended M1 path.

**MUST:** mass is **summed over layers and heads, and normalised per block.** `θ_read`'s default is consistent with this, being described as "a fifth of a block's attention".

**The problem is that the interface does not say so and nothing tests it.** `DecodeResult.attention` is typed `Mapping[int, float]` with no statement of units, so an M1 backend reporting raw attention weights, summing to the number of heads or the number of layers rather than to 1.0, satisfies the type, satisfies every test, and produces demotion behaviour wrong by orders of magnitude with **no error raised anywhere**. Integrity demotion and eviction both read this number.

This is a spec-to-code gap rather than an open design question: the answer exists in the corpus and has not been carried into the interface it constrains. It **MUST** be stated on `DecodeResult` and pinned by a conformance test before an M1 backend reports measured mass.

## 7.2 The hint channel

```
attention_hint: AttentionHint = (tags, tag_weight=0.8, recency_scale=8.0)
```

A backend that cannot measure returns `attention=None` and a hint naming source **tags**; the kernel resolves tags to segments itself (`_resolve_hint`). `AttentionHint` "exists to make the fiction visible in the type system": a hint is not a measurement, and its presence in a journal marks every conclusion downstream of it as mechanism-validation rather than evidence.

**M0:** `decode` always returns `attention=None` and a hint. So **every attention-driven behaviour observable today runs on the hint path.** The measured path in the kernel is written, and correct on inspection, but is exercised by no M0 run.

## 7.3 Masked mass is dropped, not trusted

**MUST:** the kernel discards mass reported against a segment that is unreadable or has no block in `visible(job)`, and journals `AttentionDenied`. That is the difference between hard enforcement and a request the job could decline: a backend that misreports cannot launder attention onto masked content, because the kernel checks rather than trusts.

## 7.4 `recency_profile` is dead

`ScriptedMachine.recency_profile(job, scale)` computes an exponential decay over visible blocks, normalised to 1.0. **It has no caller outside tests.** The kernel synthesises its own default in `_resolve_hint`, so the backend method is redundant. It is not on the `MachineBackend` Protocol, so no M1 backend need implement it, and it should be deleted or promoted, not left ambiguous. Recorded as OQ-AM-4.

# 8. The mask: attention as an MMU

## 8.1 What it is for

The mask is the page table. A compartment child that must not read its parent's secrets is prevented by *masking the blocks*, not by instructing the model to ignore them. ZEOS-MP's central claim, that attention masking is memory protection, reduces entirely to whether this is enforced in the backend.

```
set_mask(job, allowed_blocks: frozenset[int]) -> None
visible_blocks(job) -> frozenset[int]
```

**MUST:** content outside `visible(job)` cannot influence decoding and cannot receive attention mass (AM-I3). On real hardware this is an allowed-block bitmap applied in the attention kernel. **This is the single hardest requirement the interface places on M1**, and the one where a backend is most likely to be silently non-compliant, because a mask that is applied to logits rather than to attention, or applied per request rather than per block, will pass every test that does not specifically probe it.

## 8.2 `⊥` and `∅` are different

```
M = ⊥   ->  everything visible          (no mask set)
M = ∅   ->  nothing visible             (masked to nothing)
```

Verified: `set_mask(j, frozenset())` yields `visible = ∅`; clearing yields `visible = {0, 1}`. **MUST:** preserve the distinction. Collapsing `∅` to "unmasked" is a privilege escalation, and it is exactly the shape of the bug already found and fixed one layer up in `CapabilityTable`, where an empty table meant "unprotected" and so stripping all authority made a job omnipotent. The same trap is present here and the same discipline applies: **absence of a restriction and a restriction to nothing must never be the same value.**

## 8.3 The mask is not validated

`set_mask` accepts any set of integers. Out-of-range indices are silently dropped by the intersection in `visible_blocks`. Verified: `set_mask({99})` on a one-block context is accepted and yields `visible = ∅`.

**POLICY:** whether to reject out-of-range indices. **MUST:** if not rejected, they **MUST** be dropped rather than granted (AM-I7). The current fail-closed behaviour is correct but silent, and a mask that accidentally names no live block turns a job into one that can attend to nothing, which presents as a mysteriously unproductive job rather than as an error.

## 8.4 `clear_mask` is M0 only

`clear_mask` is **not** on the `MachineBackend` Protocol. Neither is `register_script`. Both are M0 conveniences. **MUST:** the kernel may not call them, and an M1 backend need not provide them. Setting the mask to all live blocks is the portable spelling of "clear".

# 9. Control tokens

```
TokenKind = NORMAL | CONTROL
```

Control tokens carry the kernel's own grammar: fault requests, `NEED` expressions, stub framing. ZEOS-MP requires them to be **structurally unforgeable**, meaning a job cannot fabricate one to fake a kernel message.

The mechanism is an asymmetry:

| Channel | Control tokens |
| --- | --- |
| `decode`, `allow_control=False` | **refused**, raises `ControlTokenViolation` |
| `decode`, `allow_control=True` | permitted, for the steps where the kernel wants a structured request |
| `inject` | always permitted; this is the kernel's channel |

**MUST:** enforce the `decode` restriction. **M0** enforces it by checking the kind of every token a script step emits, which makes forgery impossible by construction rather than by string matching. **MUST (M1):** enforce it at the **sampler**, by making reserved token ids unselectable. A backend that instead post-filters decoded text has not implemented this requirement; it has implemented a filter that a sufficiently determined model can be prompted around, which is the lesson the design is specifically trying not to teach.

`ControlTokenViolation` and `MaskViolation` both **signal backend bugs, not job misbehaviour.** They are not faults in the ZEOS sense, they are not routed to `on_fault`, and a job cannot provoke them by any legitimate action. Reaching either means the machine is broken and the run's results are void.

**Padding tokens are control tokens.** `pad_to_block` appends `PAD_TOKEN`, whose text is `<pad>` and whose kind is `CONTROL`. Verified. This is right: padding is kernel-introduced framing and a job must not be able to emit it.

# 10. Requests: the syscall channel

A decode step may carry one request, which is how a job asks the kernel for something.

```
MachineRequest = (op, pipe, pipes, payload, segment, resource, text, read_pipe)
OpKind = NONE | READ | WRITE | WRITE_READ | SELECT | FAULT | NEED | ACQUIRE | RELEASE | SPAWN | EXIT
```

**MUST:** at most one request per decode step.

**`WRITE_READ`** is one request that the kernel performs as two operations -- `pipe` is written, then `read_pipe` is read -- within a single `tick`, so no preemption check falls between them. It exists because the universal shape of a turn is *hand over, then sleep*, and splitting that across two ticks lets the peer the write just woke take the machine one command before the job would have yielded it anyway. Backends opt in by emitting it; one that keeps emitting separate `WRITE` and `READ` behaves exactly as before. The read half is performed only if the write half completed.

**There is deliberately no YIELD.** From the interface's own comment: *jobs cannot volunteer scheduling decisions.* Scheduling is the kernel's, entirely. A job blocks because it read an empty pipe, waited on a resource, or faulted, and never because it decided to be polite. This is the whole difference between ZEOS and an agent loop, and it is enforced by the absence of a token in an enum, which is the cheapest possible place to enforce it.

**POLICY:** how a backend extracts a request from generated text. **M0** carries it structurally in the script. **M1** will need grammar-constrained decoding -- the fourth of the four asks ZEOS makes of a serving stack. **MUST:** the request is a *request*. Every one of these is then checked by the kernel against capabilities, rings, integrity and ceilings. A backend that produces a `WRITE` has not performed a write.
