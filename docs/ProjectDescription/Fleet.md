# ZEOS-Fleet -- Teams of Robots

**Status:** design draft v0.1 (2026-08-10). Extension to the ZEOS core design.

The point of this document is that the mechanisms ZEOS already defines -- descriptors, pipes, priorities, the suspension stack, resume, and the resource table -- are **reusable**: a team of robots needs no new machinery. One shared deliberative kernel serves many bodies, and to that kernel a robot is simply a *resource a job acquires*: an embodiment is a lease (a capacity-1 resource, so blocking, priority inheritance, and deadlock detection come for free), losing a body is a suspension, and being granted a new one is an ordinary resume whose dirty diff happens to be dominated by `self.*` -- position, tooling, battery. Doorways and corridors are mutexes with capacities; a charging dock is just another lock, so fleet energy management emerges from descriptors plus the allocator rather than from a bespoke subsystem; and coupled manoeuvres (two carriers on one beam) are gangs -- co-scheduled descriptor groups dispatched and preempted all-or-none.

Because the transcript, not the machine, is the source of truth of a job, jobs are portable across bodies: a task is not "robot 7's job" but a job currently *embodied* in robot 7. The allocator that matches bodies to jobs is a swappable policy module; what the kernel owns are the invariants around it -- every grant is a revocable lease, every revocation is a suspension rather than a loss, and the lock graph over bodies and physical mutexes is global, so deadlock is detected and faulted loudly rather than timed out. **A robot is a body a job wears; changing bodies is a resume.**

```
        one mechanism                        reused as
  ┌──────────────────────┐    ┌───────────────────────────────────────┐
  │ descriptor           │───▶│ mission, handler, allocator policy,   │
  │  (one behaviour,     │    │ gang member, "backing out" behaviour  │
  │   one file)          │    ├───────────────────────────────────────┤
  │ pipe                 │───▶│ sensor feed, actuator, telemetry,     │
  │  (blocking I/O)      │    │ robot-to-fleet link traffic           │
  │ priority + stack     │───▶│ body preemption: a higher mission     │
  │  (preempt / resume)  │    │ takes the body; the loser suspends    │
  │ resume + dirty diff  │───▶│ re-embodiment: "you changed" --       │
  │  (RESUME)            │    │ self.position, self.tooling, battery  │
  │ resource table       │───▶│ embodiment lease, doorway, corridor,  │
  │  (hold/block/inherit)│    │ crane, charging dock, work cell       │
  │ children + on_fault  │───▶│ gangs (all-or-none dispatch),         │
  │  (spawn / supervise) │    │ supervision, robot-loss recovery      │
  └──────────────────────┘    └───────────────────────────────────────┘
```
