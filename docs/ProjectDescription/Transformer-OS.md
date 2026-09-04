# ZEOS -- A Transformer Operating System

**Status:** design draft v0.1 (2026-08-03). Name: "ZEOS" (Transformer Operating System).

## 1. Thesis

Complex, real-time, transformer-mediated systems should be programmed the way
real-time operating systems are programmed: as a **hierarchy of task
descriptors** scheduled by a kernel, with **interrupts** driving preemptive
switching between jobs, **pipes** providing both inter-job dataflow and
blocking-based scheduling, and a **stack** of suspended job states enabling
resumption.

The design slogan the whole system converges on:

> **Jobs block, never poll; the kernel wakes, never asks.**

Today's agent frameworks are cooperative multitasking at its worst: the agent
itself decides when to check for messages, which in LLM-land means a forward
pass per poll -- wasted tokens, wasted latency, and no bound on response time to
an urgent event. ZEOS moves blocking, waking, and preemption out of the model
and into a runtime.

### The robot example

A robot is executing a low-priority, high-level goal ("tidy the workshop").
A smoke sensor fires. The interrupt preempts the tidying job at the next token
boundary, pushes its state onto the stack, and dispatches a pinned
emergency-response handler. When the emergency handler completes, the tidying
job is popped and resumed -- with a kernel-inserted notice that the world has
changed underneath it (the robot has moved, time has passed) so it revalidates
its plan before continuing.

## 2. Core abstractions

### 2.1 Job

A job is one transformer-mediated task in flight. Its full state is:

- **Descriptor** (immutable "code"): a markdown file, §3.
- **Transcript** (portable state): the text of everything the job has read and
  generated. Survives model swaps and weight upgrades. The source of truth.
- **KV cache** (fast state): a *materialization of the transcript* for one
  specific model + weights version. Fast path for same-model resume; discarded
  or recomputed from input tokens on cross-model resume.
- **Kernel metadata**: priority, lifecycle state, held resources, pipe
  bindings, stack position, accumulated read-set, pending messages.

Descriptor + transcript + metadata = the PCB. KV is a cache of the PCB, not
part of it.

### 2.2 Priorities and the stack

- Static base priority per descriptor (integer; lower = more urgent, RTOS
  convention). Arbitrary levels -- not fixed tiers.
- Preemptive: a runnable job always preempts any running job of lower
  priority. Preempted jobs are pushed onto the **suspension stack** and
  resumed LIFO as higher-priority work drains, subject to §6 revalidation.
  Only a `READY` job preempts; a suspended job never displaces a running one,
  which is often the very handler that displaced it. But when nothing is
  running, the top of the stack competes with the ready set on priority
  (Appendix A rule 1) -- an interrupt suspends a job, it does not demote it.
- Low-priority, high-level goal jobs run only when everything above them is
  blocked or done. This is the normal idle-time behavior, not starvation --
  but see interrupt storms, §5.5.
- **Priority inheritance** on resources: a job holding a lock or lease that a
  high-priority job blocks on temporarily inherits that priority. Over pipes, which are not held -- donation follows declared relationships only:
  writers by capability, readers by stdin binding. Prevents classic priority
  inversion.

### 2.3 Interrupts

An interrupt is an asynchronous event bound to a handler descriptor via the
**interrupt vector table** (§5.1). Handling an interrupt = making the bound
handler job runnable at its priority, which (if it outranks the running job)
preempts within one token boundary.

### 2.4 Pipes

A pipe is a bounded token buffer between jobs, and simultaneously the
scheduling primitive that eliminates polling (§4).

## 3. Task descriptor format

One markdown file per descriptor. YAML frontmatter = the parts the kernel
reads; markdown body = the prompt the model reads.

```markdown
---
name: tidy-workshop
priority: 80                # lower = more urgent
model: default              # or a specific model id; affects KV portability
preemptible: true           # false only for short critical sections, §5.4
pinned: false               # true = resident handler, KV pre-filled in HBM, §5.3
budget:
  tokens: 200000            # hard generation budget; exceeding raises a fault
  deadline: none            # optional wall-clock deadline (soft-real-time jobs)
reads:                      # read-set: world state this job's plan depends on
  - robot.position
  - workshop.inventory
writes:                     # write-set: world state this job may change
  - robot.position
  - workshop.inventory
pipes:
  stdin:  user.commands     # blocking read source
  stdout: user.reports      # output sink
  tools:  robot.actuators   # tool calls are writes here; results read back
children:                   # sub-jobs this job may spawn (the hierarchy)
  - clear-bench
  - sort-fasteners
on_fault: escalate          # or: retry | abort | handler:<descriptor>
---

# Task: tidy the workshop

You are the workshop-tidying job. Your goal is ...

## Procedure
...

## On resume
If your context contains a <RESUME> notice, re-read the listed world
state before taking any physical action, and revalidate your current plan
step against it.
```

Notes:

- **Hierarchy** is expressed via `children`: a job spawns sub-jobs (which
  inherit or override priority), waits on their result pipes, and composes
  them. Writing a complex system = authoring this tree of descriptors.
- **read/write sets** are declared over a namespaced world-state vocabulary.
  They drive selective resume invalidation (§6) and resource locking. They are
  declarations of *intent*; the kernel also accumulates an observed read-set
  from what the job actually queried, and the union is used at resume time.
- **`preemptible: false`** is interrupt masking. It must be paired with a
  small token budget -- an unpreemptible job with a large budget is a design
  error the kernel rejects at load time.

## 4. Pipes

### 4.1 Semantics

- `read(pipe, n)` -- job blocks until ≥1 token/record is available; available
  content is appended to its context. Blocking **deschedules** the job: no
  forward passes, KV eligible for page-out.
- `write(pipe, tokens)` -- appends to the buffer; blocks if the buffer is full
  (backpressure). A write to an empty pipe is a **wake event** for its reader.
- Buffers are bounded, sized in tokens. Backpressure gives automatic rate
  matching between models of different speeds with zero logic in either
  descriptor.
- `select([pipes])` -- block until any of several pipes is readable. Needed for
  any job that serves multiple sources.

### 4.2 Zero-copy pipes (same model)

If producer and consumer run on the same model + weights, the pipe need not
copy tokens: the consumer attends directly over the producer's KV segment
(the RadixAttention/prefix-sharing mechanism, repurposed as shared-memory
IPC). Cross-model pipes fall back to text copy. Same distinction as shared
memory vs. sockets; the descriptor doesn't change, only the kernel's transport
choice.

### 4.3 Tool calls are pipe I/O

A tool/actuator/API call is a write to a device pipe followed by a blocking
read of the result pipe. Consequences:

- A job waiting on a slow tool is descheduled identically to one waiting on
  another job. GPU residency freed (the InferCept observation).
- The job remains preemptible while its tool call is in flight: preempt the
  job; the result waits in the pipe until resume.
- Device drivers = adapters that turn external event sources (sensors, HTTP,
  timers) into pipe writes.

### 4.4 Interrupts are pipe writes

Unification: an interrupt source is a device pipe with a handler descriptor
bound to it at high priority. A write to that pipe makes the handler runnable.
One wake mechanism serves dataflow, tool completion, and interrupts -- the
interrupt vector table is just the table of (pipe → handler, priority)
bindings.

## 5. Interrupts and preemption

### 5.1 Interrupt vector table

| field | meaning |
|---|---|
| source pipe | the device pipe whose write fires this vector |
| handler | descriptor name |
| priority | handler's dispatch priority |
| policy | `queue` (serialize repeat firings) / `coalesce` (collapse pending firings into one) / `reentrant` (spawn parallel handler instances) |
| min_interval | storm throttle, §5.5 |

### 5.2 Preemption mechanics and the latency budget

The transformer's natural quantum is the **token boundary** (one forward pass,
~ms). Preemption is therefore truly preemptive -- no cooperation from the
descriptor body -- with latency:

```
interrupt latency = (one token boundary)              ~ms
                  + (make HBM room for handler KV)     fast with paged KV; ~0 if headroom
                  + (handler context load)             ~0 if pinned (§5.3)
                  + (prefill of event payload only)    proportional to payload size
```

Target: soft-real-time in the tens of milliseconds for pinned LLM handlers.

What token-boundary preemption cannot interrupt: a physical action already
dispatched. That requires a compiled, non-LLM reflex layer below the kernel --
a separate system ZEOS does not implement.

### 5.3 Pinned handlers (resident ISRs)

High-priority handler descriptors are **pinned**: prompt prefix pre-filled at
load time, KV locked in HBM, never swapped. Dispatch cost is then only the
event-payload prefill. This is the ISR-resident-in-RAM trick; HBM pinning
budget is a first-class kernel resource.

### 5.4 Masking / critical sections

`preemptible: false` sections defer interrupt dispatch until the section's
token budget expires or it exits. Only for short atomic sequences
(mid-multi-part actuation, mid-transaction). Kernel-enforced budget cap.
Equivalent of `cli`/`sti` -- and just as dangerous, hence the cap.

### 5.5 Storms and re-entrancy

- **Coalescing**: for sensor-type sources, N pending firings collapse into one
  handler dispatch that reads the latest value (level-triggered, not
  edge-triggered).
- **Throttling**: `min_interval` per vector.
- **Starvation control**: if a low-priority job is preempted more than K times
  or starved beyond a deadline, raise a scheduler fault (visible event, not
  silent aging -- real-time systems should fail loudly).
- **Re-entrancy**: default `coalesce` or `queue`; `reentrant` only for
  handlers whose read/write sets are disjoint across instances.

## 6. Suspension, the stack, and resumption

### 6.1 Context switch = KV paging

Swap hierarchy for suspended jobs:

```
GPU HBM -- running + pinned handlers
CPU RAM -- shallow-suspended (recently preempted; likely to resume soon)
Disk -- deep-suspended (bottom of stack, long-lived goals)
```

Swap-vs-recompute per job decided by context length, depth on the stack, and
priority (the vLLM tradeoff). Cross-model resume always recomputes from
transcript.

### 6.2 Resumption is not transparent -- the one genuinely new problem

A classical OS restores registers and the process never knows it was gone.
A ZEOS job's saved state contains **beliefs about the world**, and the world
changed while handlers ran above it. Restoring KV verbatim restores *stale
beliefs that the model will keep attending to*.

Protocol on resume:

1. Kernel computes `dirty = read_set(job) ∩ ⋃ write_set(everything that ran above it)`
   (declared ∪ observed sets on both sides).
2. `dirty = ∅` → resume silently. The job's beliefs are exactly as it left
   them, so it is told nothing; the journal still records the resume and its
   duration. A notice whose whole content is "nothing changed" is not
   information, and it costs window that the pager cannot reclaim.
3. `dirty ≠ ∅` → append `<RESUME>` **carrying the diff**, e.g.:

   ```
   <RESUME>
   Suspended 94s. While suspended: handler 'smoke-response' ran.
   Changed state you depend on:
     robot.position: bench-3 → doorway
     workshop.inventory: unchanged? NO -- extinguisher removed from wall
   Revalidate your current plan step before acting.
   </RESUME>
   ```

   A bare flag is not enough -- the diff must be salient enough to override
   stale in-context state. There is exactly one resume notice, and its presence
   means something changed.

This is TLB-shootdown logic: invalidate only the lines that were touched.

An open question, testable in isolation: do current models reliably let a
`RESUME` notice override earlier in-context world state, or does this
need training?

### 6.3 Stack discipline

- Resume order is LIFO by default (interrupt semantics).
- A handler may instead **cancel** jobs below it (the emergency invalidated
  the goal) or **replace** them (dispatch a new goal job). Descriptor-level
  `on_complete` policy on the handler: `return` (default) / `cancel-below:<n>`
  / `replace-with:<descriptor>`.
- Faults (budget exceeded, tool error, deadline miss, scheduler starvation)
  are interrupts targeting the faulting job's `on_fault` policy -- same
  mechanism, so error handling is also just descriptors.

## Appendix A -- Scheduler state machine

States: `READY`, `RUNNING`, `BLOCKED` (on pipe read/write), `SUSPENDED`
(preempted, on stack), `PINNED-IDLE` (resident handler awaiting its vector),
`DONE`, `FAULTED`.

```
                         dispatch (highest-priority READY)
              READY ─────────────────────────────────────▶ RUNNING
                ▲                                            │ │ │
                │ wake: pipe readable/writable               │ │ │
             BLOCKED ◀───── read empty pipe / write full ────┘ │ │
                                                               │ │
              SUSPENDED ◀──── preempted by higher priority ────┘ │
                │  (pushed on stack; KV paged per §6.1)          │
                │ pop (LIFO) + RESUME when dirty                 │
                └────────────────▶ READY                        │
                                                                │
              DONE / FAULTED ◀──── complete / budget/deadline ──┘

  PINNED-IDLE ──vector fires──▶ READY   (payload prefill only)
  PINNED-IDLE ──write on its declared stdin──▶ READY
  FAULTED ──on_fault policy──▶ handler dispatch (same interrupt mechanism)
```

Transition rules:

1. Highest-priority job dispatches; ties broken FIFO. The candidates are the
   `READY` set **and the top of the suspension stack** -- being interrupted does
   not change a job's priority, so a suspended job is weighed against ready work
   rather than ranked behind all of it. A suspended job wins a tie: it is mid-work
   where an equal-priority ready job is between pieces of work.
2. `RUNNING` → `SUSPENDED` only via preemption; `RUNNING` → `BLOCKED` only via
   pipe ops; there is no yield -- jobs cannot volunteer scheduling decisions
   (no cooperative multitasking).
3. Wakes from `BLOCKED` go to `READY`, not `RUNNING` -- the scheduler decides.
4. Priority inheritance: a `RUNNING`/`READY` job holding a resource that a
   higher-priority `BLOCKED` job waits on inherits that priority until release.

## Appendix B -- A machine model for an LLM

A classical process is a program counter plus mutable memory, and execution
mutates memory in place. An LLM agent isn't that. The forward pass is a pure
function of the token sequence, so state *is* the sequence (we may want to add
"registers" or "slots", but that's not in current LLM architectures). The KV
cache is the process image: the materialized, resident form of that state.
Tokens are the source; KV blocks are the loaded image. There is no jump. The
program counter is always at the end.

So the kernel's entire instruction set over a context is five operations:

| Op | Meaning | Cost |
|---|---|---|
| `DECODE` | extend by model-sampled tokens | one forward pass per token |
| `INJECT(x)` | extend by externally-supplied tokens | prefill cost of size of `x` |
| `TRUNC(k)` | cut back sequence to offset `k` | free, simply drop KV blocks |
| `FORK` | duplicate a context | free, copy-on-write on KV blocks |
| `SPLICE(i,j,x)` | replace a middle span | recompute everything after `i` |

## Appendix C -- Injecting interrupt return state

Given the machine model of Appendix B, interrupts become straightforward. A model
can be interrupted at a token boundary safely for high-priority event handlers.
The following mechanisms can be used to inject the return state of an interrupt
into a task/turn.

- **Flags.** The interrupt sets a flag in the job's bookkeeping state and waits
  for the agent to query the flag. Context is unaffected.
- **Sampler bias.** The LLM sampler is changed to nudge token probabilities
  towards some outcome (e.g. "check messages").
- **Status region.** Small section within the context window that can be
  rewritten in place.
- **Append.** Insert a message at the next block boundary. Wrap in start
  `<MESSAGE>` and end `</MESSAGE>` tags. The start tag can optionally include an
  identifier. Multiple messages may be appended if they have accumulated between
  suspension and resumption of a job.
- **Preempt.** Terminate the current iteration and insert the message
  immediately.
