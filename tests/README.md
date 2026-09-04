# The test suite

Four tiers:

| Tier | Question it answers |
| --- | --- |
| `unit/` | Does one component do what its docstring says? One behaviour against scripted pipes; the serde tests walk the event registry, so a new event is covered automatically. |
| `contract/` | Do the interfaces hold both ways? Pipe roles, the injection corpus against MP faults, the machine backend contract (parameterised over backends -- adding a real backend means adding one entry to the list, not writing a suite), and policy-as-test rules like `test_core_is_stdlib_only.py`. |
| `integration/` | Do subtrees compose? Assertions are made against **journal properties** -- "the alarm preempted supervision within one boundary" -- never against transcripts. |
| `replay/` | Did the *shape* of a run change? The golden traces below. |

Run everything with `uv run pytest`; the determinism gate alone with
`uv run pytest -m determinism`. Markers are strict (`--strict-markers`): a new
marker must be declared in `pyproject.toml`, and the vocabulary grows
deliberately as CI lanes appear.

**All tests pass** There is no flaky-test allowance, no retry, and no
quarantine list. The kernel is deterministic by construction, so a test that
fails intermittently is a bug in the kernel or the test.

## The golden traces

`replay/golden/` is the project's golden-value regression suite. Each trace
hashes the **event-kind sequence** of a reference run -- not the full journal,
because full-journal hashes break on every reworded fault detail, and a fixture
that cries wolf gets regenerated reflexively, which destroys its value. The
kind sequence changes only when the *shape* of the run changes: a new
preemption, a lost fault, a different eviction.

If a golden test fails, first decide whether the shape change is intended. If
it is, the **only legitimate way** to refresh the fixtures is:

```bash
uv run python -m tests.replay.regenerate
```

then read the diff of the regenerated trace and say in the commit message why
the shape moved.
