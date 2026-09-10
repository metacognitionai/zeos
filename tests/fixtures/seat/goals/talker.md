---
name: talker
priority: 50
pipes:
  stdout: ops.report
script:
  - emit: "say hello there;"
  - emit: "write stdout all done;"
  - emit: "exit;"
---

# Task: say one thing, report, and finish

You write commands, one at a time, each ending in a semicolon. Say something, then
write your report to stdout, then exit.
