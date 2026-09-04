---
name: counter2
priority: 100
script:
  - emit: "one"
  - emit: "two"
  - emit: "three"
  - emit: "four"
  - emit: "five"
  - spawn: counter2
  - exit: true
---

Count from one to five, one number per turn. At five the count is finished:
SPAWN, then EXIT.
