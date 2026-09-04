# Demos

Demonstrations built on the zeos kernel live here. Each has its own `pyproject.toml` and
is a member of the repository's uv workspace, so `uv sync --all-packages` at the root
installs the kernel and every demo into one venv, and a clone is enough to run one.

| | |
|---|---|
| [`counter/`](counter/) | The smallest thing the kernel will run: jobs that count to five, spawn a successor and exit. Works with a scripted tape, the Claude API, and a local llama.cpp Qwen. A kernel journal can be produced with or without a model in the loop. |
| [`space-invaders/`](space-invaders/) | A real-time game a served model is too slow to play. A deliberating job asks the model and is preempted mid-sentence by a reflex that needs no forward pass, then resumes with the kernel's diff of what moved. Three drivers over one game — a human, a prompt loop, a descriptor tree — so the comparison is one architecture against another over a fixed world, and the case states its own success criteria for the journal to be judged against. |
| [`coop-count/`](coop-count/) | A real `MachineBackend` over llama.cpp, and two counting agents that hand a baton back and forth. The long-form tutorial builds the backend from nothing, and the same application appears twice — once driven by blocking pipe reads, once by vectors. |

A demo carries its own cases. The kernel does not: see the note in the top-level README
about why no application-specific case ships in `src/`.
