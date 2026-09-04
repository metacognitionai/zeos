# space-invaders

A real-time game a served model is too slow to play, and three ways to drive it:
a human at a keyboard, a prompt loop, and a two-job descriptor tree the ZEOS
kernel schedules. One package, one `pyproject.toml`, managed with uv.

This file is the handoff: layout, invariants, and what is unfinished.
`README.md` is the user-facing guide and is the place for anything a user needs.
Keep them apart.

A demo in this tree is self-contained and depends on the kernel by relative path
(`demo/README.md`). Nothing here imports anything outside the package except
`zeos` itself; nothing in `src/zeos/` imports this.

## Layout

The files at the top of the package are the entry points; everything else is in
a package of its own. `src/zeos_space_invaders/*.py` is the list of things you
can run, and a subdirectory is a job the package does.

| | |
| --- | --- |
| `play.py`, `cli.py`, `compare.py`, `viewer.py` | the four entry points, one per console script. Nothing else lives at this level. |
| `game/` | the rules (`rules.py` -- `Rules` is what game this is, and `Game` reads nothing else), the stick every driver writes to (`controls.py`), where a shot connects (`aim.py`), the `reset`/`step` wrapper and `snapshot` (`env.py`, which carries the live rules downstream). Stdlib only, and it knows nothing about models. |
| `players/` | who is answering. `base.py` holds `PromptPlayer` -- the prompt, the parser, `respond`, the lazy `sdk()` loader -- and `rules_prompt`, the rules text both arms are given; `random_player.py`, `openai_compat.py`, `claude.py` are the three backends over it. `endpoints.py` resolves a vendor's model, endpoint and key once, for the players and the machines alike; `sampling.py` is the one table of effort levels and sampling parameters; `rules_text.py` fills the prompt's numbers from `Rules`. |
| `players/prompts/` | one shared rules body plus one decision section per view. Templates, not prose: every number is a `{placeholder}` filled by `rules_text.fields`. Read only by `base.py`. Editing the wording needs no code change; adding a number needs a field. |
| `players/zeos/` | `api_machine.py` is the machine behind the kernel's seam, with `api_openai.py` and `api_claude.py` as the two wire formats over it; `player.py` is the native reflex, the sensors, the device adapter and `build_kernel`. The program they run is in `cases/`. |
| `cases/space-invaders/` | the program: three pipes, four world objects, one vector, two descriptors and the criteria a run is judged against, all YAML and markdown in the flat shape `zeos.descriptor.load_case` reads. |
| `clocks/` | whether the world waits. `step.py` waits for the agent, `realtime.py` does not, and `zeos.py` is the realtime clock with the kernel deciding rather than a prompt loop; `outcome.py` is the half of a summary all three answer the same way. |
| `web/` | serving a run as a page: the projection (`payload.py`), the server (`server.py`), the page (`static/`, with the one vendored asset under `static/vendor/`). Imports zeos, because the kernel lane is `zeos.monitor`'s fold; nothing that plays a game imports it. |
| `runlog/` | what a run leaves behind: record shapes (`schema.py`), the directory writer (`writer.py`), the reader everything else joins through (`reader.py`). Stdlib only, and it knows nothing about players or clocks. |
| `utils/` | `views.py` (how the state is shown -- the single biggest lever on whether a model can play), `config.py` (`.env` loading and `--settings` files) and `flags.py` (the board flags and the `Rules` they build, shared by `play`, `agent` and `compare`). Only what more than one area needs. |

`tests/` mirrors that layout: one file per area, plus `stubs.py` (the fake SDK
clients) and `conftest.py` (fixtures). Nothing imports one test module from
another.

`players/prompts/` and `cases/` live *inside* the package because
`packages = ["src/zeos_space_invaders"]` ships everything under it; a case or a
prompt resolves from `__file__`, so an installed copy behaves like a checkout.

## What a run writes

One directory per run, the same shape whoever played. `runlog/` owns it and
every entry point goes through it.

```
runs/<stamp>-<player>-<view>-<effort>-seed<n>/
    meta.json      the request: settings resolved, prompt in full, commit
    events.jsonl   frames and decisions, interleaved
    kernel.jsonl   zeos only
    summary.json   the outcome; its existence means the run finished
```

- **`meta.json` is the request, `summary.json` is the outcome, and neither
  repeats the other.** `RunReader.row()` is the single place they are joined,
  and every table is built from that.
- **The prompt states the game that will run, and states it once.** Nothing in
  `players/prompts/` or `utils/views.py` writes a board dimension, a monster
  count, a life total or a bomb rate as a literal: `rules_text.fields` fills them
  from the `Rules` of the run, and `views.describe` renders the grid and the
  worked examples out of a real game played under those rules. A new rule the
  model needs to know becomes a field and a placeholder.
- **A vendor's endpoint is resolved in one place**, `players/endpoints.py`, for
  the prompt-loop player and the machine both, so the two arms of a comparison
  cannot pick different models out of one shell. Likewise `sampling.py` for
  effort and sampling: a run labelled `--effort none` is the same request on
  both arms.
- **The half of a summary every clock shares lives in `clocks/outcome.py`.** What
  each runner still says itself is what it means differently: `unparseable`,
  where `usage` comes from, and the word for a run cut short.
- **A frame per world tick, written by the clock, not by the player.** Frames
  carry positions, not the drawn board; the rendered prompt stays on the
  decision, because that is what the model saw.
- **The clock owns tick numbers, the player owns content.** `Decision.tick` is
  the tick whose frame the chooser looked at; `tick_applied` is where the action
  landed. The zeos pilot is stamped with the tick it was *asked* on, several
  ticks before it answers. `RunWriter.decision` refuses a record with no tick.
- **The machine owns the completion, and the driver owns the tick.**
  `players/zeos/api_machine.py` is a real `MachineBackend`; `decode` is where a
  piece of the reply becomes a token the kernel can account for. `ZeosDriver` is
  the device adapter -- sensors into pipes, pump, actuator, world. So a
  scheduled `Decision` carries what the kernel said (who wrote the stick, whether
  a preemption landed) and the model's numbers reach `summary.json` through the
  runner: `usage`, `decoded_words`, `thinking_words`, `generations`,
  `cancellations`, `voided`.
- **Two counters that must not be merged.** `preemptions` is counted off
  `kernel.jsonl` (`job.preempted`) and is the claim; `cancellations` and `voided`
  are the machine's own account of I/O it stopped wanting. Likewise `usage` is
  what the server reported -- nothing for a completion the kernel cancelled --
  and `decoded_words` is what arrived on the wire.
- **`kernel.jsonl` is a real zeos journal**, written with `zeos.journal.codec`,
  so `zeos inspect`, `zeos debug` and `zeos.monitor.fold` read it without
  knowing this game exists. The one addition is `tick` on each line, which joins
  a kernel event to a frame; the envelope is built in `ZeosDriver.journal()`
  because `runlog` imports nothing but the standard library. `meta["case"]` is
  the repo-relative case path so `zeos debug` can draw the wiring, and
  `meta["zeos"]` is the kernel commit the journal came from.
- **The kernel lane does not grow into a debugger.** The lane is one
  `SystemView` per world tick trimmed to 200 bytes, the only thing that can join
  a kernel event to a board. Ordering inside a tick and the case's bindings are
  `zeos debug`'s.
- **A comparison is a run of runs**: `runs/<stamp>-compare/episodes/<player>-seed<n>/`
  is a plain run directory and `RunReader` recomputes `table.json`. In the viewer
  the comparison is the head of its episodes and carries the means
  (`payload._rollup`).
- **`actions_per_tick` means two things** and `RunReader.row()` merges them: in
  `meta.json` the cap asked for, in `summary.json` the rate achieved. A column
  has to pick one on purpose.

## Two rules that look like bugs and are not

- **A monster that drops onto a missile's square is not hit; the two pass
  through each other.** `Game.tick` moves and tests the missile *before* the
  block marches. `game/aim.py` models it; changing it would invalidate every
  published score.
- **Pressing into a wall, and firing with a missile already up, are no-ops** --
  and they are the only ways to spend a turn without moving. The game has no
  wait action. The reflex relies on this: standing still is written as `shoot`.

## Environment

uv, one venv, Python 3.12 (the floor zeos sets, pinned in `.python-version`).

```bash
uv sync                    # the game, the tests, and neither vendor SDK
uv sync --extra openai     # plus the OpenAI SDK
uv sync --extra claude     # plus the Anthropic SDK
uv run pytest -q                       # with coverage; the floor is 90%
uv run pytest -q --no-cov tests/x.py   # one file, while iterating
```

Coverage is measured on every run and the floor is enforced (`fail_under = 90`).
Running one file fails the floor; pass `--no-cov`. If a change drops it, the
answer is the missing test, not a smaller number.

`anthropic` and `openai` are extras named after the `--player` they serve and
are imported inside the player that needs them via `base.sdk()`, never at module
import. `uv run play`, `--player random` and the whole test suite run with
neither installed. A run that needs one exits with the `uv sync --extra` line
that fixes it.

## Running things

Four console scripts: `uv run play`, `uv run agent`, `uv run compare`,
`uv run viewer`. Usage is in README.md; the invariants:

- Two backends by design, one per wire protocol: the OpenAI SDK on the
  **Responses** API (never chat completions) for the prompt loop, and the
  Anthropic SDK. The machine behind the kernel uses chat completions, because
  the Responses API cannot continue a partial assistant turn. Adding a vendor
  means a third backend, not a framework.
- Config resolves `.env` < environment < flags. Env vars are the two SDKs' own
  names (`OPENAI_*`, `ANTHROPIC_*`), so a Claude Code session's
  `ANTHROPIC_BASE_URL` is followed. The banner prints where each run points.
- `--settings` reads a JSON file of those same flags in a `board` object and a
  `play` object, placed before the command line so a flag typed there still
  wins. `agent` takes both objects, `play` takes `board` alone, so a person and
  a model can be handed the same game: `settings_default.json` is the one
  README.md runs, `settings_ablation.json` the one `mini_ablation.md` runs.
- **`--effort none` is checked, not trusted.** If reasoning tokens come back the
  run raises `ThinkingNotDisabled` on turn one. Do not weaken that to get a run
  moving: a server that ignores the switch scores whatever a truncated reply
  parses as, which is indistinguishable from a bad model. Omitting a switch is
  not turning it off -- Claude thinks by default, so `--effort none` sends
  `thinking: {"type": "disabled"}` explicitly.
- `--clock realtime` does not wait for the agent, so latency costs ticks;
  `--clock step` is the deterministic one. `zeos-*` players are realtime-only and
  refuse `--clock step`: nothing is ever late there and nothing can be preempted.
- **`compare` runs one episode at a time.** Under `--clock realtime` the latency
  is the measurement, and concurrent episodes would share the endpoint.
- **Every game rule is a `Rules` field, and every run records what it was.**
  `Rules` is a frozen dataclass whose defaults are the module constants; `Game`
  reads its own copy and nothing else. Downstream reads the live rules off
  `snapshot(game)["rules"]`, never the module -- a view or a reflex answering
  from the defaults while the game is 5x5 is wrong in the direction that
  matters. `turns_to_land` and `turns_to_reach` are methods on `Rules` with no
  module-level counterpart for that reason, and `Rules.__post_init__` refuses a
  board that cannot hold the game described.
- `outcome` is `lost` both when the ship is shot down and when the block reaches
  the bottom; `lives` distinguishes them, and `hits_standing` /
  `hits_walking_in` say how each life went -- a bomb landing on a ship already in
  its column, or on one that stepped into the column that tick.
- `MAX_STEPS` is one number in `game/env.py`; `agent` and `compare` both default
  to it.
- **One write, one move; no write, no move.** The rule lives in
  `game/controls.py` and every driver goes through it -- human, prompt loop,
  kernel. Do not reintroduce a "hold the last action" path: it emits moves nobody
  chose and lets a run where the model never answered score.
- `--actions-per-tick` is the stick's budget and applies to every player alike.
  Slowing `--tick` for a slow agent speeds the player up relative to its own
  missile; pair it with `--actions-per-tick 1`.

## The aim point

`LeadView` hands the model the one calculation it cannot do without reasoning
tokens: where a shot connects. `game/aim.py` finds it by playing the rules
forward, breadth-first over the three actions, with `Game` itself doing the
stepping so the search cannot drift from the rules. Two things it must keep:

- **The bombs already in the air are obstacles.** New fire is random and is not
  rolled; the bombs already falling are deterministic, `_restore` keeps them, and
  a step that ends under one is not a line. Without this the aim point sends the
  player into a bomb landing next turn while the "incoming fire in your column"
  cue truthfully says none.
- **The view prints the first move of the line, not a direction to the column.**
  The two differ whenever the line begins with a step aside or a turn standing
  still, and printing the direction sent the player into the bomb the search had
  just avoided.

## The kernel side

- **The driver must not drain the kernel before it delivers a sensor reading.**
  `ZeosDriver` hands over the model's pieces first, pumps only until some job is
  running, and delivers the threat into that. A driver that runs to quiescence
  before each event only ever delivers interrupts to an idle kernel, which is
  the one case where the interrupt does not matter. `_pump(until_running=True)`
  is the whole of it.
- **A scheduled run reads its reply in pieces, structurally.** Preemption needs
  the pilot to be *running* when the interrupt lands. The machine hands over one
  word per `decode`, and when the queue is dry `decode` returns `tokens=()` --
  the empty step ZEOS-AM §6.1 permits -- which is the only shape that leaves the
  job running while the model thinks. `--no-stream` means nothing for `zeos-*`.
- **The threat sensor watches the ship's column two ticks out and the two
  columns beside it one tick out.** The pilot's move lands two or three ticks
  after the board it was chosen against, so a bomb landing next tick one column
  over is a threat to a move already in flight. The reading names what is clear
  -- a side, or `stay` when the ship's own column is safe -- and `dodge` writes
  `shoot` for `stay`. Firing the vector preempts the pilot and cancels its
  stale completion, which is the protection.
- **Three pipes, all of them the game's.** `game.state` and `game.threats` are
  sensors, `game.controls` is the actuator. `game.state` is level-triggered: the
  adapter replaces an unread reading rather than queueing behind it. How a token
  reaches this process is not part of the game and gets no pipe.
- **The world holds game state, and the stick's readback is not game state.**
  `ship.column`, `block.front_row` and `missile.in_flight` are what a resumed job
  is told changed. An object earns its place by passing three tests at once:
  someone else can change it while the job is suspended, the change would make
  the job choose differently, and it fits on one line -- monster positions fail
  the last two and would make every resume dirty. `game.action` stays declared
  but **no job may read it**: its `world_object` is what makes `game.controls`
  latch rather than queue (without it a job that actuates once a turn backs up
  behind its own writes), and a diff over it says what someone did rather than
  what is now true -- a dodge steering the way the stick already pointed is a
  same-valued write, dropped, and the pilot would resume clean over the top of
  it.
- **The driver publishes the world every tick and applies the stick inside the
  pump.** Publishing unconditionally is safe because `WorldStore.set` drops an
  idempotent write. Applying inside the pump is not an optimisation: the kernel
  resumes the pilot in the same breath the handler exits, so a dodge applied at
  the end of the tick has not moved the ship when `dirty_for` runs.
- **A clean resume is not "nothing happened", and the machine cannot see one.**
  A same-valued write produces an empty diff and an empty diff injects nothing,
  so the machine learns nothing. `ZeosDriver._invalidate_preempted` reads
  `job.preempted` off the journal and calls `APIMachineBase.invalidate` for each
  job named. The driver has to do it: neither the kernel nor the machine tells
  the other about a preemption, and jobs do not volunteer scheduling facts.
- **The tail of a cancelled completion is marked `VOID`, never deleted.** The
  kernel has already charged those tokens and extended the segment table over
  them; shortening `T` behind it is `trunc`, which is the kernel's to call. The
  offsets the machine keeps (`first_offset`, `committed`) move with every
  splice, as ZEOS-AM §6.5 requires. `partial` picks what is voided: `syscall`
  (the structured default) keeps everything up to the last completed call,
  `keep` marks nothing, `drop` marks the whole completion.
- **What a job read is remembered, not inferred from its context length.**
  `APIMachineBase.inject` records arrivals for a natively-served job and
  `_decode_native` drains them into `Native.arrived`. A mark held at the context
  length breaks the first time the pager shrinks it.
- **One thread per completion and no pool.** The machine opens at most one
  completion per context; `stranded` counts cancelled producers still running.
  The client carries a timeout and `_cancel` never waits.
- **The virtual clock is the driver's, not a wall clock.** `ZeosDriver._pump`
  calls `kernel.advance_time` once per token boundary at `NS_PER_TICK`
  (`players/zeos/player.py`), so `deadline: 5ms` means five token boundaries and
  a `latency` criterion is reproducible on any hardware. `_pump`'s `max_ticks`
  counts boundaries, not seconds; the paced runner passes a `time.monotonic`
  deadline instead, and every batch runs at least one boundary.
- **The kernel boots before the clock thread starts, and the boot is journalled
  as tick 0.** `ZeosRealtimeRunner.run` calls `driver.start()` and flushes its
  journal before pacing begins; booting inside the first batch stamped
  `kernel.started` with whatever tick a 1ms clock had reached by then.
- **The driver publishes the world before the first pump, coalesces boards, and
  never delivers a threat and a board on the same tick.** `sense` reads an
  unread board out of `game.state` before writing the new one, so a pilot
  slower than the world wakes to the newest board rather than the oldest; and
  the stick is last-write-wins, so a pilot move chosen against a board delivered
  alongside a threat would land after the dodge and undo it.
- **The case judges itself.** `cases/space-invaders/criteria.yaml` holds the
  capability claims in zeos's own vocabulary, evaluated off the journal by
  `zeos.demo.criteria` and reported in `summary["criteria"]`. Written against
  source pipes and world objects, never a descriptor name. The
  `resumed_dirty_naming` criterion needs a dodge that moved the ship; an episode
  in which every dodge stood still fails it, truthfully.
- **The pilot is booted; only the threat has a vector.** A vector consumes its
  source payload and injects it, so a vector-spawned pilot blocks on a pipe the
  kernel has already drained. `boot.yaml` names the pilot and `ZeosDriver.start`
  spawns it. Do not add a vector for a level-triggered sensor a job reads in a
  loop.
- **A handler does not read its own stdin.** `evade` gets the threat with the
  dispatch and writes at its first token boundary. Adding the read back blocks
  it forever.
- **The pilot plays by the prompt loop's rules.** `goals/pilot.md` is what is the
  pilot's own -- a reply is a syscall, what a resume notice means, that something
  faster watches the fire -- and `build_kernel` appends `players.base.rules_prompt`
  for the `Rules` and view of the run. What still differs between the arms
  besides scheduling: history is the prompt loop's `deque(maxlen=5)` against the
  pilot's paged transcript, and the reply is read whole against a piece at a
  time. The claim is architecture against architecture, judged on the same
  runlog.

## The machine

`players/zeos/api_machine.py` turns a streaming HTTP completion into decode
steps. What it must keep:

- **Only a narrowing mask cancels a generation.** The kernel refreshes the mask
  at every block boundary and a decoding job widens it each time; treating any
  change as a context change would restart the reply every block. Padding does
  not cancel either: pads are never transmitted, so the request is unchanged.
- **`_runs` splits the transcript into speaker runs by origin.** A pad, void or
  masked span breaks a run, or the descriptor body and a pipe arrival would
  merge into one system message; `THINK` does not break one, being the same
  speaker. Framing is dropped by origin, never by kind, so a `CONTROL` token
  cannot be forged from text and a native behaviour is checked for one.
- **End of turn is the machine's.** When the stream ends `decode` itself issues
  `read stdin`; a model-issued read only yields, because `game.state` is
  level-triggered, and a prompt left to end its own turn wrote without reading.
- **The `text` format's clause grammar**: a clause begins at a verb in `VERBS`
  and ends at the next semicolon, because `tokens_from_text` strips newlines
  from the body; the half-built clause of a cancelled generation is reset before
  the next one starts, or the two are spliced into one write.
- **The JSON schema is one `move` enum**, with op and pipe supplied by the
  machine because each has one legal value, and no `steps` array because the
  server does not enforce `minItems`. A malformed object means the server did
  not enforce the schema, not that a request was made. A content schema also
  suppresses in-band `<think>`, so `reasoning_effort` is the one effort switch
  that reaches a served Qwen. There is no tool-calling format: the server parses
  tool calls after the fact rather than constraining them.
- **The vendor seam is two threads.** `_request` runs on the kernel's thread and
  is the only thing that reads `ctx`; `_stream` runs on the producer thread and
  publishes `gen.wire` while open so `_abort` can close the socket from the
  kernel's thread -- a one-request-at-a-time server otherwise queues the next
  generation behind an abandoned one. `continuing` a partial assistant turn is
  vendor-specific: vLLM and SGLang take `continue_final_message`, an OpenAI
  endpoint rejects the field, Anthropic refuses a request not ending in a user
  turn.
- **`stall_s` is the preemption granularity and the per-tick cost.** The driver
  runs up to 64 boundaries a tick, so at 5ms a tick could cost 320ms; at 1ms the
  realtime arm holds its declared tick. Round trip and TTFT are measured in the
  machine from `_Generation.opened`, the same span the prompt loop calls
  latency; a cancelled generation contributes nothing.
- **Claude specifics.** Thinking arrives as a summary, so `budget.tokens` counts
  what the model described rather than what it spent. The cache breakpoint goes
  on the `system` block, the one part of the context that does not change; the
  CLI's `effort="none"` is translated to thinking off, because Anthropic's
  `output_config.effort` has no such level.

## The viewer

The index and comparison screens are Tabulator tables; the run screen's kernel
lane is the hand-rolled `table()` in `viewer.js`.

- **The library is vendored, not linked.** `web/static/vendor/tabulator/` holds
  the two dist files and `vendor/README.md` records the version and the edits.
  `viewer --export` has to open from a filesystem with no network, so
  `grep -c 'url(' tabulator.min.css` stays `0`. An exported run does not get it:
  `page(vendor=False)` empties the `/*VENDOR_CSS*/` and `/*VENDOR_JS*/` markers.
- **The columns are declared once, in `viewer.js`.** A derived number belongs in
  `payload.py`, because the page holds no arithmetic and has no tests. `LAYOUT`'s
  version suffix has to be bumped when `COLUMNS` changes: Tabulator stores a
  layout per browser under that id and a stored layout beats the code's
  defaults forever. Widths are not persisted, so the measured `minWidth` can
  take effect.
- **`minWidth` is measured** by `floorFor` in an offscreen probe sharing one
  type declaration with the header (`.probe` in `viewer.css`). `min` on a
  descriptor raises the floor where the data is the wide thing; `grow` gives it
  more of any spare width.
- **A reload is `replaceData` inside `keepingPlace`, never a rebuild**, so sort,
  filters, chosen columns and scroll position survive. `reloader` says how a
  screen reloads itself or that it cannot; `renderRun` sets it to null. The
  timer does not stop on `document.hidden`: an embedded pane can report hidden
  while on screen.
- **A Tabulator has to be destroyed, not dropped.** `dropTables` does it on
  every screen change. A column menu lives on `document.body`, and one left
  behind matches `refreshNow`'s open-menu guard forever.
- **`diff` is a second table and persists nothing.** `persistence: false` is the
  load-bearing line; a persisting table would store the hidden columns as the
  reader's own choice. It compares drawn text (`textOf`, `outcomeText`), the
  picks are keyed by run path in `picked`, and `restoring` keeps a `replaceData`
  wipe from reading as the reader unticking everything.
- **The index opens `meta.json` and `summary.json` and nothing else.** `aimed`
  reads every decision and frame and belongs on the comparison's episode rows,
  not on an index that reloads itself every few seconds.
- **`SCHEMA_VERSION` in `runlog/schema.py` names the run-directory format**;
  `RunReader` refuses a directory written under another. The `NEWLINE` marker
  that carries a board through a pipe saves the rows but not the padding that
  aligned the columns, so a `grid` row reaches the pilot as nine words.

## How this hangs off the kernel

```toml
[tool.uv.sources]
zeos = { workspace = true }
```

The demo is a member of the repository's uv workspace, like `demo/counter` and
`demo/coop-count`: one venv and one lockfile at the root, and `uv sync` there
installs the kernel alone -- this package comes in with `uv sync --all-packages`,
`--package zeos-space-invaders`, or a `uv sync` run inside this directory. The
kernel is installed editable, so `kernel_version()` reports the checkout it used
-- `direct_url.json`'s commit or `git rev-parse`, with `-dirty` when the tree has
uncommitted work. The case lives inside the package, unlike `demo/coop-count`,
so `agent` runs with no `--case` and survives being installed as a wheel.

## Not done yet

- **The descriptors are called `pilot` and `evade`**; zeos's own README calls
  them `play` and `incoming-fire`. `Decision.by` carries the names into every
  run log, so rename them together with the logs, not before.
- **An environment world write is not journalled.** The driver publishes
  `ship.column` and the rest with `world.set` directly; zeos has no public "the
  environment observed a new value" entry point, and `accept_replica` is
  federation's. The effect is visible in `job.resumed`'s `dirty` list.
- **A scheduled `Decision` carries no prompt or reply.** The machine keeps no
  per-completion record, so `prompt`, `reply`, `reasoning` and `usage` are empty
  for `zeos-*` and the viewer's per-decision panes are blank for a scheduled
  run. It would need a per-generation record surviving retirement, read by the
  driver when it builds the `Decision`.
- **Three declared deviations from ZEOS-AM.** `_runs` is a second page table the
  kernel cannot see: padding, reasoning and voided output are resident, unmasked
  and not transmitted. AM-I6 cannot hold: `stall_s` is a wall clock, so the
  journal's `virtual_ns` differs run to run and a run is not replayable from its
  journal, though `T` itself is timing-independent. And a transport error
  reaches the kernel as an exception, because `OpKind` has no operation for a
  machine to raise a `TOOL_ERROR` fault.
- **The pager does not yet hold the scheduled arm's request size flat**, so its
  time to first token rises with episode length. Root's own note on `stub_size`
  says why.
- **`zeos-claude` has run single episodes only**; every published scheduled
  number is `zeos-openai`. Claude's effort levels are barely explored: Sonnet 5
  takes `thinking.type.adaptive` plus `output_config.effort`, and a summarised
  thinking block makes `budget.tokens` count the summary rather than the work.

## Testing

The suite runs with **neither vendor SDK installed**, against fake clients in
`tests/stubs.py`: no key, no server, no network. A test that needs
`uv sync --extra` is a test nobody runs.

- `tests/stubs.py` owns the fakes, `conftest.py` the fixtures (`run` and
  `written`, an open run directory and its reader; `no_dotenv`, autouse, which
  keeps the checkout's `.env` out of every test).
- Both entry points are driven end to end: `test_cli.py` runs `agent` including
  the scheduled path (`cli` -> driver -> kernel -> runner) with a stub-backed
  player, and `test_play.py` runs the human loop against a fake curses screen.
- The refusals are tested as carefully as the runs: `--clock step` under a
  scheduled player, a `zeos-` player passed to `compare`, and a server that
  reasons when told not to each have a message of their own and a test asserting
  it.
- `test_aim.py` drives whole games by the search alone and insists they win, and
  that following the line never walks into a bomb; a recommendation that misses
  or steps under fire shows up there and nowhere else.
