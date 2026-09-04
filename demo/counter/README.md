# counter -- one job, counting to five on repeat

The smallest thing the kernel will run. The `simple_count` case boots two identical
descriptors, `simple_count/goals/counter{1,2}.md`,
whose script emits the words one to five, spawns a fresh instance of itself, and
exits. The restart is a **new job**, not a loop: a script is a finite list of steps,
so "on repeat" means `spawn` then `exit`.

To install ZEOS and run the example:

```bash
# 1. Install (needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh)
git clone git@github.com:metacognitionai/zeos.git
cd zeos
uv sync --all-packages          # the kernel, plus this demo's two model seats

# 2. The scripted seat. Nothing else needed: no key, no weights, no network.
uv run python demo/counter/run.py --case simple_count

# 3. The Claude seat. Needs a key in the environment.
export ANTHROPIC_API_KEY=...
uv run python demo/counter/run.py --case simple_count --machine claude

# 4. The same model through a claude login instead of a key. One
#    `claude --print` per move; unset the key first, or the CLI prefers it.
claude login                    # once per machine
unset ANTHROPIC_API_KEY
uv run python demo/counter/run.py --case simple_count --machine claude-code

# 5. The local seat. Fetch the weights once (~2.7GB, Qwen3.5-4B Q4_K_M),
#    then run: llama.cpp on this machine's CPU, no key and no network.
uv run python demo/counter/fetch_model.py
uv run python demo/counter/run.py --case simple_count --machine qwen
```

Every run prints one line per journal event, and the text of every decode beneath
it. All four seats will hopefully (due to non-determinism) have the same output:

```
  t0    kernel.started block_size=16 case=simple_count
  t0    descriptor.loaded descriptor=counter1 priority=100 placement=any preemptible=True
  t0    descriptor.loaded descriptor=counter2 priority=100 placement=any preemptible=True
  t0    job.spawned job=1 descriptor=counter1 priority=100 owner=kernel integrity=2
  t0    job.spawned job=2 descriptor=counter2 priority=100 owner=kernel integrity=2
  t0    job.state job=1 from_state=ready to_state=running
  t0    job.dispatched job=1 priority=100
  t0    machine.inject job=1 segment=1 pipe=kernel principal=kernel ring=1 integrity=1 tokens=18 text=[18 items]
  t0    segment.opened job=1 segment=1 ring=1 integrity=1 perms=Perm.R|X|P
  t0    segment.closed job=1 segment=1 end=18
  t0    mask.updated job=1 allowed_blocks=2
  t1    machine.decode job=1 segment=2 tokens=1 text=one
  t1      decoded "one"
  t1    machine.block_boundary job=1 block=1
  t1    vm.working_set job=1 size_tokens=19 segments=2
  t1    mask.updated job=1 allowed_blocks=2
  t2    machine.decode job=1 segment=3 tokens=1 text=two
  t2      decoded "two"
  t3    machine.decode job=1 segment=3 tokens=1 text=three
  t3      decoded "three"
  t4    machine.decode job=1 segment=3 tokens=1 text=four
  t4      decoded "four"
  t5    machine.decode job=1 segment=3 tokens=1 text=five
  t5      decoded "five"
  t6    job.spawned job=3 descriptor=counter1 priority=100 parent=1 owner=kernel integrity=2
  t7    job.state job=1 from_state=running to_state=done
  t7    job.completed job=1 tokens_used=5
  t8    job.state job=2 from_state=ready to_state=running
  t8    job.dispatched job=2 priority=100
  t8    machine.inject job=2 segment=4 pipe=kernel principal=kernel ring=1 integrity=1 tokens=18 text=[18 items]
  t8    segment.opened job=2 segment=4 ring=1 integrity=1 perms=Perm.R|X|P
  t8    segment.closed job=2 segment=4 end=18
  t8    mask.updated job=2 allowed_blocks=2
  t9    machine.decode job=2 segment=5 tokens=1 text=one
  t9      decoded "one"
  t9    machine.block_boundary job=2 block=1
  t9    vm.working_set job=2 size_tokens=19 segments=2
  t9    mask.updated job=2 allowed_blocks=2
  t10   machine.decode job=2 segment=6 tokens=1 text=two
  t10     decoded "two"
  t11   machine.decode job=2 segment=6 tokens=1 text=three
  t11     decoded "three"
  t12   machine.decode job=2 segment=6 tokens=1 text=four
  t12     decoded "four"
  t13   machine.decode job=2 segment=6 tokens=1 text=five
  t13     decoded "five"
  t14   job.spawned job=4 descriptor=counter2 priority=100 parent=2 owner=kernel integrity=2
  t15   job.state job=2 from_state=running to_state=done
  t15   job.completed job=2 tokens_used=5
```

## What each line means

`t0`, `t1`, ... is the tick. One tick is one turn, and a job says at most one token
per turn.

| line | meaning |
| --- | --- |
| `kernel.started` | the kernel booted. Context is handled in blocks of 16 tokens |
| `descriptor.loaded` | a descriptor file was read and can now be run |
| `job.spawned` | a job was created from a descriptor. `parent=` means another job asked for it |
| `job.state` | the job changed state: ready, running, or done |
| `job.dispatched` | the scheduler gave this job the machine |
| `machine.inject` | text was put into the job's context from outside. Here, its goal |
| `segment.opened` | those tokens became a segment. `perms` says what may be done with them |
| `segment.closed` | that segment is finished. New tokens go in a new one |
| `mask.updated` | the kernel set which blocks the job is allowed to look at |
| `machine.decode` | the job said something. This is the only line where a token is produced |
| `decoded "..."` | the words from that decode, printed by `run.py` |
| `machine.block_boundary` | the job filled a block. Housekeeping happens here, not every token |
| `vm.working_set` | how big the job's context is right now |
| `job.completed` | the job is done. `tokens_used` counts only what it said, so spawn and exit do not count |

## Debugger

There is also a debugger. Save the run to a journal file:

```
uv run python demo/counter/run.py --case simple_count --journal demo/counter/runs/counter.jsonl
```

and then view it in the browser at `localhost:8000`:

```
uv run zeos debug demo/counter/simple_count --journal demo/counter/runs/counter.jsonl
```

**Tip.** You can append the `--no-open` flag if you already have the debugger open in your browser or are port forwarding from a different machine.
