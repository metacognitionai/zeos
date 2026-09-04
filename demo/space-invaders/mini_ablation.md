# Ablation setup

A 9×8 ASCII board, a real clock, and two ways to drive it: a model in a prompt
loop, or the ZEOS kernel scheduling that same model as a preemptible job beside
a reflex that needs no forward pass.

## The experiments

One model (`qwen/qwen3.6-35b-a3b`, served locally), the 9x8 board and the 0.2s
tick of `settings_ablation.json`, one view (`lead`), `--effort none`, five
seeds, run one arm after another.

| | lives left | monsters killed | seconds per tick |
| --- | --- | --- | --- |
| ① `openai --clock step` -- the world waits for the model | 2.8 | **9.8** | 0.48 |
| ② `openai` -- prompt loop, clock running | 1.0 | 3.4 | 0.21 |
| ③ `zeos-openai` -- the same model and rules, scheduled | **3.0** | **6.0** | 0.20 |

Row ① is the model with the world stopped while it thinks: a tick there is one
request, 0.48s on average. Rows ② and ③ face a clock that ticks every 0.2s
whether or not the model has answered. The prompt loop's move is 0.49s old when
it lands -- 2.4 ticks -- so it is shot down standing still, or walking into a
bomb that was not there on the board it answered. The scheduled arm loses no
lives at all: a reflex job outranks the pilot, and when a bomb is about to land
on the ship or beside it, the kernel takes the pilot off the machine mid-reply
and the reflex gets clear, 67 times over these five episodes. With
`qwen/qwen3-4b` the rows read 2.8 / 9.8, 0.6 / 4.8 and 3.0 / 6.0.

## How to run them

```bash
# 1. Install uv, then the demo. From demo/space-invaders/ in a ZEOS checkout
#    (or `uv sync --all-packages --extra openai` from the checkout root):
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra openai          # add --extra claude for the Claude players

# 2. Point it at a model. The variable names are the two SDKs' own, and any
#    flag on the command line wins over the file.
cp .env.example .env            # OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL

# 3. Row ②: the model plays, one request per move, against the running clock.
uv run agent --settings settings_ablation.json --player openai --seed 7

# 4. Row ③: the same model, scheduled by the kernel as a job a reflex can preempt.
uv run agent --settings settings_ablation.json --player zeos-openai --seed 7

# 5. Watch either run back: the board, the move that landed on each tick and,
#    on the scheduled run, what the kernel was doing, on one scrubber.
uv run viewer                   # browse runs/ at http://127.0.0.1:8123

# 6. Read the kernel itself -- the case's wiring and every journal event in
#    order -- with the repository's debugger, from the ZEOS checkout root:
cd ../.. && uv run zeos debug demo/space-invaders/src/zeos_space_invaders/cases/space-invaders \
  --journal demo/space-invaders/runs/<id>/kernel.jsonl
```

`--player openai` works with vLLM, SGLang, Ollama, LM Studio and hosted
providers alike: they all serve the Responses API. `--view lead` is the view that
wins, `--effort none` asks for no reasoning and checks that none came back, and
`--clock step` plays row ① by making the world wait for the model.
