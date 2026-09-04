---
name: supervision
priority: 80
reads:
  - plant.unit_a
writes:
  - plant.schedule
pipes:
  stdout: ops.report
script:
  - emit: "reviewing the production schedule for this shift"
  - emit: "unit throughput within tolerance"
  - emit: "checking auxiliary unit duty"
  - emit: "schedule confirmed for the shift"
  - write: {pipe: ops.report, text: "shift schedule confirmed"}
  - exit: true
---

# Task: supervise production for this shift

You are the production supervision job. Review the shift schedule against current
plant state and confirm or revise it.

## On resume

If your context contains a `<RESUME>` notice, re-read the listed plant state
before confirming anything -- the plant may have been reconfigured underneath you.
