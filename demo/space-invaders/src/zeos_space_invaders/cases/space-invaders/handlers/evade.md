---
name: evade
priority: 5
pinned: true
preemptible: false
budget:
  tokens: 8
# No `reads:`: the threat arrives with the dispatch, and a pinned, unpreemptible
# handler is never suspended, so there is no diff for it to be told about.
writes:
  - game.action
pipes:
  stdin: game.threats
capabilities:
  - pipe: game.controls
    min_integrity: 2
---

# get out of the way

Fire is about to land in your column, or in the column beside you that a move
already on its way would step into. Get clear: off your column if it is yours,
still if it is the neighbour's.

The reading is already in front of you — the kernel consumes the payload that
fired this vector and injects it as it dispatches you, so there is nothing to
read first. Name a clear side and write it to `game.controls`, in one step.

You are pinned, unpreemptible and given eight tokens, because this must complete
inside one world tick. There is nothing to deliberate about: the reading names
what is clear -- a side to step to, or `stay` when your own column is safe and
the bomb is landing next door -- and any of them beats the move that was
already on its way.

This handler is deterministic code behind the machine seam, not a served model —
a forward pass costs four world ticks and would arrive after the hit. That is the
whole point of it being an interrupt.
