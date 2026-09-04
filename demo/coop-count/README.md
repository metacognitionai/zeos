# coop-count — a real model under ZEOS

A demo in the ZEOS repository, and a member of its uv workspace, so it shares the
kernel's venv and lockfile and runs from a checkout with no published release.

```bash
# 1. Install (needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh)
git clone git@github.com:metacognitionai/zeos.git
cd zeos
uv sync --all-packages       # one venv for the kernel and both demos
cd demo/coop-count

# 2. Run the scripted case. No weights, no key, no network, and it ends on its own.
uv run zeos-count run cases/coop-count-scripted --machine scripted \
  --events cases/coop-count-scripted/events.jsonl

# 3. Run it on the Claude API. No weights, no GPU, one call per command.
export ANTHROPIC_API_KEY=...
uv run zeos-count run cases/coop-count-pipe --machine claude --max-ticks 40

# 4. Or the same model through a claude login instead of a key. One
#    `claude --print` per command; unset the key first, or the CLI prefers it.
claude login                 # once per machine
unset ANTHROPIC_API_KEY
uv run zeos-count run cases/coop-count-pipe --machine claude-code \
  --seat-model opus --max-ticks 40

# 5. Or fetch local weights instead (~2.7GB, Qwen3.5-4B Q4_K_M).
uv run zeos-count fetch-model

# 6. And run on those. Ctrl-C to stop; it does not end on its own.
uv run zeos-count run cases/coop-count-pipe --journal out.jsonl --threads 32

# 7. Press space while it runs, type a number, press Enter.
#    A keypress is a pipe write, the pipe is bound in the vector table, and the
#    handler that fires outranks both counters.
```

In every case `--machine` picks what answers the model's turn. Output looks like this if
the keyboard doesn't interrupt:
```
counter-a  say 1
counter-a  say 2
counter-a  say 3
counter-a  say 4
counter-a  say 5
counter-a  say 6
counter-a  say 7
counter-a  say 8
counter-a  say 9
counter-a  say 10
counter-a  ──▶ count.progress_a 10
counter-a  ──▶ count.a2b    go
counter-a  ... waiting on count.b2a
counter-b  say 11
counter-b  say 12
counter-b  say 13
counter-b  say 14
counter-b  say 15
counter-b  say 16
counter-b  say 17
counter-b  say 18
counter-b  say 19
counter-b  say 20
counter-b  ──▶ count.progress_b 20
counter-a  ◀── <STATUS count.b> 20 </STATUS>
counter-b  ──▶ count.b2a    go
counter-b  ... waiting on count.a2b
counter-a  ◀── go
counter-a  say 21
counter-a  say 22
counter-a  say 23
counter-a  say 24
```

**Two handover architectures, one job to do.** Both count upward for ever, ten at a
time, two agents taking turns, and neither agent has a loop in it. They differ in how a
turn hands over.

`cases/coop-count-pipe/` shows **waiting**. Each agent reads a pipe when its turn ends,
which puts it to sleep until its peer writes, so a job that is waiting runs no forward
passes at all and costs nothing. Nobody polls: the taking of turns *is* the two jobs
blocking and waking.

`cases/coop-count-vector/` shows **starting**. There a write does not wake the peer but
creates one: each agent counts its stretch, records it, starts the next agent, and
exits. Nothing has to be held ready at boot, and it pays for the tidiness by re-reading
the whole task each turn.

Neither agent remembers anything. The count goes out to the world and comes back as one
line the kernel keeps up to date, so they keep counting no matter how much of their own
working the kernel throws away.

`cases/coop-count-scripted/` shows **the same interrupt without a model**. It is
`coop-count-pipe` with a tape added: each descriptor's `script:` block lists the
commands that job issues, `--machine scripted` plays them, and `events.jsonl` presses
the key.

## Interrupting

Press **space** and an interrupt takes the machine off whichever job is running: a
keypress becomes a pipe write, the pipe is bound in the vector table, and the handler
that fires outranks both counters. Type a number, press Enter, and it goes into world
state; the job that was suspended resumes with a `RESUME` notice naming what
changed, and restarts its segment from there rather than from what it remembers.

## Paging

One thing to know before you swap the model: ZEOS pages contexts, paging is SPLICE, and
SPLICE needs a KV cache you can truncate. Qwen3.5's SSM layers do not have one, so every
splice re-prefills the whole context — `LlamaMachine._rewind` handles it, at a cost.
Qwen3.5 is the faster decoder in exchange. A plain transformer such as
`bartowski/Qwen2.5-7B-Instruct-GGUF` pages far more cheaply and runs with no other
change.

See [TUTORIAL.md](TUTORIAL.md) for the full walkthrough.
