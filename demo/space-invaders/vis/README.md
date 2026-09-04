# vis

Draws a GIF of a `demo/space-invaders` game.

## Running it

```bash
python demo/space-invaders/vis/run.py --live       # watch it play, no GIF
python demo/space-invaders/vis/run.py              # play a game, then draw it
python demo/space-invaders/vis/run.py --draw-only  # redraw what is there
```

Playing needs `ANTHROPIC_API_KEY`, which `demo/space-invaders/.env` supplies;
drawing needs neither a key nor a network.

`--set section.key=value` changes one setting for one run and leaves the file
alone.

Two run directories drawn side by side:

```bash
cd demo/space-invaders/vis
uv run --with pillow python compare_gif.py \
  runs/zeos-haiku-w12 runs/claude-prompt-loop-w12-haiku \
  side-by-side-w12-haiku.gif --duration 500
```

## settings.json

| line | what it is |
| --- | --- |
| `run` | directory the run is written to and drawn from |
| `gif` | file the GIF is written to |
| `width`, `height` | the board, in squares |
| `monster_rows`, `monster_cols` | how many monsters, as a block |
| `monster_col_offset` | column the block starts at |
| `lives` | hits the ship survives |
| `missile_rows` | rows the ship's shot climbs per tick |
| `danger_rows` | rows a bomb falls per tick |
| `march_group` | monsters that move per tick |
| `fire_chance` | chance a monster drops a bomb on any tick |
| `player` | `zeos-claude`, `claude`, or `random` |
| `model` | model id, or `null` for the demo's default |
| `view` | how the board is written for the model |
| `effort` | thinking depth, or `none` for no thinking |
| `seed` | fixes the random stream |
| `tick` | seconds per world tick |
| `actions_per_tick` | cap on actions landing in one tick |
| `max_steps` | episode cap |
| `history` | previous boards the model is shown |
| `cell` | pixels per board square |
| `duration` | milliseconds per drawn frame |
| `scale` | multiplies the finished image |

`duration` is per frame, not per world tick: these runs used `tick: 0.5`, so
`500` is real time and the `80` default plays at 6x.
