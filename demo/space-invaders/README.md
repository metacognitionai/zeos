# space-invaders -- a model under a clock that does not wait

A 12×16 ASCII board, a real clock, and three ways to drive it: you at the
keyboard, a model in a prompt loop, or the ZEOS kernel scheduling that same
model as a preemptible job beside a reflex that needs no forward pass.

## Get started

```bash
# 1. Install uv, then the demo. From demo/space-invaders/ in a ZEOS checkout
#    (or `uv sync --all-packages --extra claude` from the checkout root):
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra claude          # add --extra openai for the OpenAI players

# 2. Create a .env file with your Anthropic API key. See .env.example for the
#    other variables it reads.
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 3. Play it yourself: arrows or a/d to move, space to shoot. The same board
#    and seed the model gets below, so the two runs sit side by side.
uv run play --settings settings_default.json --seed 7

# 4. Let the model play, one request per move, against the running clock. The
#    board is drawn in the terminal as it goes. settings_default.json is the
#    game: a 12x16 board in its `board` object, and how the model is asked -- a
#    0.5s tick, one action per tick, seed 7 -- in its `play` object.
uv run agent --settings settings_default.json --player claude --out runs/prompt-loop

# 5. The same model, scheduled by the kernel as a job a reflex can preempt.
uv run agent --settings settings_default.json --player zeos-claude --out runs/scheduled

# 6. Watch either run back: the board, the move that landed on each tick and,
#    on the scheduled run, what the kernel was doing, on one scrubber.
uv run viewer                   # browse runs/ at http://127.0.0.1:8123

# 7. Read the kernel itself -- the case's wiring and every journal event in
#    order -- with the repository's debugger, from the ZEOS checkout root:
cd ../.. && uv run zeos debug demo/space-invaders/src/zeos_space_invaders/cases/space-invaders \
  --journal demo/space-invaders/runs/scheduled/kernel.jsonl
```

## How it plays

A world tick is half a second and a request takes a few seconds, so the model is
structurally four or five ticks late and no prompt fixes that. The scheduled
player is therefore two jobs at different priorities. The `pilot` reads the
board and asks the model, streaming the reply back **a piece at a time** so its
decode becomes token boundaries the scheduler can take away. Above it sits
`evade`, pinned at a higher priority and bound to a threat sensor through the
interrupt vector table: when fire is about to land on the ship, or in the column
beside it that a move already on its way would step into, it takes the pilot
off the machine mid-reply and gets clear, in microseconds, because it is
arithmetic and not a model call. The run log records which of the two moved on
every tick.

## What a run writes

```
runs/20260827-153854-openai-lead-none-seed7/
    meta.json      what was asked for: every resolved setting, the prompt in full
    events.jsonl   the run itself: a `frame` per world tick, a `decision` per move
    kernel.jsonl   zeos players only: the kernel's own journal, keyed by world tick
    summary.json   how it turned out: the verdict, the aggregates, the criteria
```

A `decision` carries the prompt sent, the reply, the reasoning, the usage, who
decided (`human`, `random`, `model`, or the zeos jobs `pilot` / `evade`), the
tick it looked at and the tick its action landed on. `kernel.jsonl` is a real
zeos journal -- `zeos inspect`, `zeos debug` and `zeos.monitor` read it directly.

## Flags

| flag | default | what it does |
| ---- | ------- | ------------ |
| `--settings` | none | a JSON file of these same flags, underscores for hyphens, in a `board` and a `play` object; the command line wins over it. `play` reads `board` alone |
| `--player` | `random` | `openai` / `claude` for the prompt loop, `zeos-openai` / `zeos-claude` for the same backend scheduled by the kernel |
| `--view` | `lead` | what the model is shown |
| `--clock` | `realtime` | or `step` for one action per tick |
| `--tick` | 0.2 | seconds per world tick in realtime mode |
| `--effort` | the server's default | how hard the model reasons; `none` is checked, not trusted |
| `--seed` | random | fixes the episode's random stream |

`uv run agent --help` has the rest.

## The ablation

`mini_ablation.md` measures the two arms over five seeds on the 9x8 board of
`settings_ablation.json`. The prompt loop answers a board that has moved on, so
it is shot down standing still or walking into a bomb that was not there when
it decided; the scheduled arm is taken off the machine mid-reply and gets clear.
