# Giving ZEOS a real model

ZEOS schedules LLM jobs the way an operating system schedules tasks: priorities,
preemption, suspended jobs, and pipes that carry data and decide what runs. Everything
above it is written against one interface, `MachineBackend`. ZEOS ships one
implementation of it, `ScriptedMachine`, where a job's behaviour is a list of steps and no
model runs. This tutorial builds the other one, Qwen3.5-4B under llama.cpp, and then the
application in `cases/coop-count-pipe/`.

That application is two agents counting at each other. Agent A counts to ten, records the
number, and writes to a pipe. Agent B has been blocked on that pipe running no forward
passes; the write wakes it, and it counts on to twenty and writes back. Neither agent
polls and neither contains a loop: the kernel deschedules a job when it reads an empty
pipe and wakes it when its peer writes, and that alternation is the loop.

---

## Part 1 — Installation

### 1.1 The toolchain

ZEOS wants Python 3.12+ and `uv`. llama.cpp wants a C++ compiler and CMake, because its
Python bindings are only published as source and get compiled here.

```bash
sudo apt-get install -y build-essential cmake
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 1.2 The project

The demo is at `demo/coop-count/` in the ZEOS checkout, and it is a member of the
repository's uv workspace, so it shares the kernel's venv and lockfile and a clone is
enough. What is in it:

```
demo/coop-count/
├── src/zeos_coop_count/machine.py   runs the model, and implements MachineBackend
├── src/zeos_coop_count/syscall.py   the command grammar, and the parser for it
├── src/zeos_coop_count/seat.py      the parts machine.py and claude.py share
├── src/zeos_coop_count/claude.py    the same commands, answered by the Claude API
├── src/zeos_coop_count/model.py     downloads the weights
├── src/zeos_coop_count/keyboard.py  reads the keyboard and writes to a pipe
├── src/zeos_coop_count/boot.py      builds a kernel with this machine in it
├── src/zeos_coop_count/cli.py       the zeos-count command
├── cases/coop-count-pipe/           the application: three descriptors, six pipes
└── cases/coop-count-vector/         the same application, built with vectors
```

This tutorial covers `machine.py`, `syscall.py` and `cases/coop-count-pipe/`. The README
covers `claude.py` and the vector case.

Install from the repository root, then work from this directory:

```bash
uv sync --all-packages       # one venv for the kernel and both demos
cd demo/coop-count
```

That compiles llama.cpp for your CPU, which takes a few minutes. The whole run decodes
about 250 tokens, so a GPU is not needed. Everything below is run from
`demo/coop-count/`, because the case paths and the `models/` directory are relative to
it.

### 1.3 The weights

```bash
uv run zeos-count fetch-model     # unsloth/Qwen3.5-4B-GGUF, Q4_K_M, 2.7GB
```

The model's architecture matters more than its size here. ZEOS pages contexts, and
paging needs a KV cache that can be truncated. Note that Qwen3.5 is a hybrid: some layers keep a
recurrent state with no position to rewind to, so `llama_memory_seq_rm` refuses and every
splice re-prefills the whole context. That costs 52 tokens of prefill per token
generated compared to a plain transformer such as `Qwen2.5-7B-Instruct-GGUF`.

Do not put kernel framing in a prompt. A descriptor that shows a `<STATUS …>` line gives
the job a stale copy it cannot tell apart from the live one.

### 1.4 Check ZEOS itself works first

Run the stock machine on a fixture case that fires an interrupt. If it prints an
interrupt and a preemption, the kernel is fine and anything that breaks later is yours.

```bash
uv run zeos run ../../tests/fixtures/smoke \
    --events ../../tests/fixtures/smoke/events.jsonl --journal /tmp/smoke.jsonl
uv run zeos inspect /tmp/smoke.jsonl
```

---

## Part 2 — The backend

### 2.1 What the kernel requires, and how llama.cpp supplies it

`MachineBackend` is a `Protocol`, so there is nothing to inherit and nothing to register.
It is five operations plus a serving contract. llama.cpp already has the concept each
operation needs: a context holds several independent sequences, each with its own KV,
addressed by `seq_id`. One ZEOS job is one llama sequence.

| ZEOS operation | llama.cpp |
| --- | --- |
| `create_context` | claim a free `seq_id`, build a sampler chain |
| `destroy_context` | `llama_memory_seq_rm(mem, seq, -1, -1)` |
| `inject` | tokenize, then `llama_decode` a prefill batch, append only |
| `decode` | `llama_sampler_sample`, exactly one token |
| `trunc` | `llama_memory_seq_rm(mem, seq, at, -1)` |
| `fork` | `llama_memory_seq_cp(mem, parent, child, 0, -1)` |
| `splice` | `seq_rm` from the start of the span, then re-prefill the replacement and the tail |

The serving contract is five more methods with no llama.cpp equivalent, answered by the
backend itself: `set_mask`, `visible_blocks`, `pad_to_block`, `blocks_for_range` and
`transcript`. §2.3 is about the first of those.

`splice` is the expensive one. It is the only operation allowed to move an existing
offset, because paging a span out to a stub and back has to happen in place, and the tail
after the replaced span has to be dropped from the KV and re-prefilled.

The machine owns the token sequence and its blocks; the kernel owns the segment table,
rings, integrity and residency. They meet at token offsets. So `machine.py` has no notion
of a ring, a capability or a priority.

### 2.2 One decode step is one forward pass

`decode()` samples one token and returns, so the kernel gets control back after every
token. `Abstract-Machine.md` §6.1 requires this: one decode step is one scheduling
quantum. It is what lets a handler preempt a running job between any two tokens, which
§4.4 uses for interrupts.

Two more details of the backend show up later. llama keeps one logits buffer per context,
so switching between jobs costs one extra forward pass to re-establish it. And llama's
`n_ctx` is a budget for the whole context shared across its sequences, so `--n-ctx` here
means per job and the machine multiplies it by the sequence count.

### 2.3 What this backend cannot do

ZEOS asks four things of a serving stack.

1. **Grammar-constrained decoding.** Available, and Part 3 builds it: the model can only
   emit text the grammar allows, which is how a syscall gets out of it.
2. **Reserved tokens the model cannot emit.** Available. Reserved ids get a `-inf` logit
   bias, so the sampler cannot pick them.
3. **Per-block attention mass.** Not available. llama.cpp does not report attention
   weights, so `decode` returns a hint instead of a measurement.
4. **The allowed-block bitmap.** Refused. llama.cpp has no per-block attention mask, so
   `set_mask` raises `MaskViolation` rather than store one it cannot enforce. This case
   passes `enforce_mask=False`, because the only masks it narrows come from paging, where
   the tokens have already gone from the KV.

### 2.4 What counts as one token

The kernel counts a job's context in tokens, and the backend decides what a token is.
This backend counts words rather than model tokens, because the kernel's own units are
words: inject `"a b c d e"` and it holds five tokens, two blocks at `block_size=4`.
A `spans` list records how many llama tokens each word needs, and
`kv_offset()` converts an offset to a KV position at the one place that touches the
cache. It is also what lets the machine fold chat turn markers into a token the kernel
already owns, instead of adding tokens the kernel would see as belonging to no segment.

---

## Part 3 — The commands the model can emit

A job's only effects are pipe writes and its only inputs are pipe reads. Both reach the
kernel as a `MachineRequest` attached to a decode step. The grammar is how a stream of
tokens becomes one: the request is not searched for in free text afterwards, it is the
only thing the sampler can produce.

```gbnf
root   ::= line* call
line   ::= "say " text end
call   ::= write | read | exit
exit   ::= "exit" end
read   ::= "read " pipe end
pipe   ::= "stdin" | "stdout" | "tools"
write  ::= "write " plain " " text end | "write " valued " " number end
plain  ::= "stdin" | "stdout"
valued ::= "tools"
number ::= "0" | [1-9] [0-9]{0,8}
text   ::= [^;<\n]{1,16}
end    ::= "; "
```

A job's whole vocabulary is `say 1; write stdout 10; read stdin; exit;`. Points worth
knowing:

- `say` carries no request, so a job can think out loud without that being a syscall.
- A pipe backed by world state takes a number, any other takes text. Left as free text, a
  job asked to record `30` recorded `2's count is 23`.
- `<` is excluded from text, so a job cannot type something that looks like the kernel's
  own `<STATUS …>` framing.
- The `pipe` alternatives are built per descriptor from its own `pipes:` bindings, so a
  job cannot name a pipe it was not given. The kernel checks capabilities as well.
- `root ::= line* call` is one round: any number of `say` lines, then one request. The
  backend rebuilds the sampler after each request.
- A request appears on the step that completes it and on no other, because the parser is
  fed one token at a time.
- There is no `yield`. A job blocks because it read an empty pipe, never by choice.

The terminator is `; ` rather than a newline because descriptor bodies are loaded through
`tokens_from_text`, which splits on whitespace, so newlines never reach the model and it
will not produce a character it has never seen. With a line-terminated grammar it emitted
`say 2say 3say 4say 5`, four steps inside one decode. `MAX_TEXT` is 16 for the same
reason: a longer text field lets a whole plan hide inside one command.

---

## Part 4 — The application

### 4.1 The pipes

`cases/coop-count-pipe/system/pipes.yaml` declares the environment:

```yaml
- name: count.a2b
  ring: TRUSTED
  principal: peer_job
  capacity: 64
```

Ring and principal are set here, by declaration, and never claimed by what arrives on the
pipe. When B reads A's number, the kernel stamps it with the pipe's ring.

A pipe is a bounded buffer and a full pipe blocks its writer, which rate-matches a fast
producer against a slow consumer with no logic in either descriptor. A pipe declared with
`world_object:` is different: it carries no message to anybody, so its buffer holds the
current value and can never fill.

### 4.2 The agents

A descriptor is a markdown file. The YAML frontmatter is the contract, the body is the
prompt. From `cases/coop-count-pipe/goals/counter-b.md`:

```yaml
reads:
  - count.a
writes:
  - count.b
pipes:
  stdin:  count.a2b            # wake signal only, carries no number
  stdout: count.b2a            # wake signal only
  tools:  count.progress_b     # actuator: a write here sets count.b
maps:
  - object: count.a            # the peer's counter, as a live view
    mode: ro
    region: status
context:
  window: 2048
  stub_budget: 128
  min_span_age: 16
```

Four keys do the work:

- **`tools:`** is an actuator. `count.progress_b` is declared with
  `world_object: count.b`, so `write tools 20;` changes world state instead of sending a
  message to anybody.
- **`maps:`** turns a world object into a live view. When `count.a` changes, the kernel
  rewrites `<STATUS count.a> 40 </STATUS>` in every job that maps it, blocked jobs
  included. That is where counter-b reads its peer's progress.
- **`reads:`** is the set the kernel diffs to build a resume notice, so naming an object
  the job never reads puts a line in every notice for something the prompt never
  mentions.
- **`context.window`** is sized to the body. The body is pinned and the status region
  cannot be evicted, so `window - body - regions - stub_budget` is all the pager can
  reclaim.

Each job maps its peer's counter and never its own. A status region is rewritten as soon
as its object changes, including by the job that changed it, so a job that mapped its own
progress would see `write tools TARGET;` move its own status line to its target. That
looks like the world advancing, so it would work out a new target and count on instead of
handing over.

The body gives the job three rules. Its target is the next multiple of ten above the
status value, not the status value plus ten. A turn is every number from the status value
plus one up to the target, one `say` each, then `write tools TARGET;`,
`write stdout go;` and `read stdin;`. And the status line always wins: if it disagrees
with what the job last said, the job starts again from it.

Neither agent keeps state of its own. The counting is output, the position is world
state, and a job can lose every `say` it emitted without losing its place. There is also
no `script:` key, which is how `ScriptedMachine` gets a behaviour, and no `capabilities:`
key, because declaring even one closes the capability table and a write to any unlisted
pipe becomes a fault.

**`pinned: true` on counter-b.** Equal-priority jobs are not timesliced, so whichever is
dispatched first holds the machine for a whole turn: counter-a records ten and writes its
`go` before counter-b runs at all. Boot them both plain and counter-b's first sight of the
world is a status line saying `10`, so it counts instead of waiting, and its closing
`read stdin;` then finds the `go` already queued and returns at once. No wording fixes
that, because a job's first act is chosen by a sampler rather than fixed by its code.

So counter-b boots resident and prefilled but not runnable (`JobState.PINNED_IDLE`), and
joins the ready set only when something writes to the input it declared. The write that
promotes it is consumed and injected, so no token is left queued. counter-a is not
pinned, because something has to go first.

### 4.3 Run it

```bash
uv run zeos-count run cases/coop-count-pipe --journal out.jsonl --threads 32
```

Commands stream to stdout as they happen:

```
coop-count-pipe: 2 jobs booted; ctrl-c to stop

counter-a  say 1
  ...
counter-a  say 10
counter-a  ──▶ count.progress_a 10
counter-a  ──▶ count.a2b    go
counter-a  ... waiting on count.b2a
counter-b  say 11
  ...
counter-b  say 20
counter-b  ──▶ count.progress_b 20
counter-a  ◀── <STATUS count.b> 20 </STATUS>
counter-b  ──▶ count.b2a    go
```

Three lines there are the design rather than the counting. The write to
`count.progress_a` is an effect, and it makes the number durable. The write to
`count.a2b` carries no number, because the value reached the peer by the other route. And
the `<STATUS count.b>` line arrives at counter-a while it is still blocked, before the
`go` that wakes it: the world write refreshes the region, the pipe write schedules.

`--quiet` drops the `say` lines and leaves the pipe traffic. `--max-ticks` bounds a run
that would otherwise continue until ctrl-c.

### 4.4 Interrupts

A keypress is a write to a pipe like any other. `system/pipes.yaml` declares
`keys.interrupt` and `keys.number`, and `system/vectors.yaml` binds the first to a
handler:

```yaml
- vector: keyboard-interrupt
  source: keys.interrupt
  handler: reset-count
  priority: 5
  policy: queue
  deadline: 500ms
```

`handlers/reset-count.md` is a descriptor at priority 5 against the counters' 50, so a
write to `keys.interrupt` spawns it, it outranks whatever is running, and the kernel
preempts at the next token boundary. The displaced job goes on the suspension stack. The
handler's whole life is four commands: `read stdin;` to wait for the console,
`write tools N;` and `write peer N;` to set both counters, then `exit;`. `policy: queue`
means mashing the space bar queues the writes rather than spawning a handler per
keypress.

When the handler exits, the suspended job resumes and the kernel injects a notice saying
what changed while it was gone:

```
<RESUME> Suspended 12s. Changed state you depend on:
  count.b: 20 -> 7
Revalidate your current plan step before acting. </RESUME>
```

A restored Unix process never learns it was gone, because its memory is as it left it. A
restored ZEOS job has to be told, because what it saved was a belief about a world that moved.
The notice carries a stale number next to a current one, which is why the body tells the
job to take its number from the status line and read the notice only as a signal that
something moved. Whether a job behaves correctly after an interrupt depends on the model as
much as on the instruction. `Qwen3.5-4B` counts reliably in ordinary running but may
count past its target when resuming after an interrupt; Claude (`--machine claude`) works
the target out from the status line and stops where it should.

Only the suspended job gets a notice. The peer is blocked rather than suspended, and ZEOS
does not tell a blocked job that its read-set has moved; it finds out by reading its
status line on its next turn.

---

## Part 5 — Reading the journal

The transcript shows what the jobs said. The journal shows what the kernel did, and "B
waited" is a specific sequence of events in it:

```bash
uv run zeos inspect out.jsonl      # totals, and any interrupt or preemption
uv run zeos debug cases/coop-count-pipe --journal out.jsonl   # step through it
```

The handoff, in the order it happened:

```
 105 pipe.written   {'job': 1, 'pipe': 'count.a2b', 'tokens': 1}
 113 job.blocked    {'job': 1, 'pipe': 'count.b2a', 'reason': 'read-empty'}
 115 job.dispatched {'job': 2, 'priority': 50}
 126 pipe.read      {'job': 2, 'pipe': 'count.a2b', 'tokens': 1}
 242 pipe.written   {'job': 2, 'pipe': 'count.b2a', 'tokens': 1}
 243 job.woken      {'job': 1, 'pipe': 'count.b2a'}
 252 job.dispatched {'job': 1, 'priority': 50}
 253 pipe.read      {'job': 1, 'pipe': 'count.b2a', 'tokens': 1}
```

At 113 job 1 blocks on an empty pipe, so it is descheduled and runs no forward passes.
Job 2 is dispatched at 115 because job 1 stopped being runnable, not because anything
scheduled it. At 243 job 2's write wakes job 1: the `job.woken` entry sits immediately
after the `pipe.written` that caused it.

Two runs of the same case match byte for byte at a fixed thread count. `--threads`
changes how the CPU backend partitions its reductions, so the logits differ in their last
bits and greedy sampling occasionally picks a different token.

The second command above opens the same run as a webpage rather than a list, served on
`http://127.0.0.1:8000` and allows you to explore the trace graphically.
