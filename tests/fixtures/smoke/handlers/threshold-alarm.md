---
name: threshold-alarm
priority: 5
pinned: true
preemptible: false
budget:
  tokens: 64
writes:
  - plant.unit_a
pipes:
  tools: actuators.unit_a
script:
  - emit: "threshold alarm acknowledged"
  - write: {pipe: actuators.unit_a, text: "max"}
  - exit: true
---

# Task: respond to a threshold alarm

A monitored reading has crossed its alarm threshold. Take the unit to maximum and
confirm.

Do not diagnose, do not report, do not plan. Take it to maximum and return.
