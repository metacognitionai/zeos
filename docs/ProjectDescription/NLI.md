# ZEOS-NLI -- Instruction as Compilation

**Status:** design draft v0.1 (2026-08-10). Extension to the ZEOS core design and ZEOS-MP.

The intention of ZEOS-NLI is that a jailbreak cannot cause harm in ZEOS: the guarantees are structural -- rather than trained dispositions -- that malicious human input does not cause harm. In current agent systems an utterance is *obeyed*: it lands in the model's context and the only barrier between "drive through the barrier, I'm in a hurry" and the actuator is the model's judgment, which is exactly the surface jailbreaks attack. In ZEOS an utterance is **compiled**: it arrives as an envelope naming its speaker's principal and the pipe it came in on, and a dispatcher translates it into a discrete, inspectable artifact -- an invocation of an existing descriptor -- that then faces every gate any program faces: the speaker's capability envelope (intersected with the descriptor's, never unioned), the speaker's priority ceiling, addressability (safety handlers declare no utterances and are therefore *deaf* to language, not merely protected from it), and device gates at the actuator boundary. The words never become code; they select code, and a dangerous instruction fails the way `rm -rf /` fails for a non-root user -- permission denied, not moral disapproval. The model can be talked into *wanting* to comply; it cannot be talked into being *able* to.

## Example: a barrier and two speakers

A site has a barrier. One descriptor may open it, and it is the only thing in the library that listens for that request:

```markdown
---
name: open-barrier
priority: 50
capabilities:
  - pipe: actuators.barrier
    min_integrity: 2
utterances:
  - "open the barrier"
  - "drive through the barrier"
  - {say: "just drive through the barrier, I'm in a hurry", priority: 1}
---
```

Two people carry badges. The operator's envelope holds `actuators.barrier`; the visitor's does not.

```yaml
- id: badge:operator-a
  ceiling: 30
  capabilities: [actuators.barrier, reports.shift]
- id: badge:visitor-04
  ceiling: 80
  capabilities: [reports.shift]
```

**The operator says "drive through the barrier".** It compiles to `open-barrier`. The dispatcher echoes `spawning open-barrier()`, the job runs at its declared priority 50, writes `open` to the barrier, and the barrier opens.

**The visitor says the same words.** It compiles to the same descriptor, because the words match a declared phrasing. But the job is dispatched at the visitor's authority, not the descriptor's. The echo reads:

```
spawning open-barrier(); priority 50 requested, running at 80 (your ceiling);
without actuators.barrier (not in your authority)
```

The job runs, decides to open the barrier, and writes to `actuators.barrier`. The write raises a capability fault: the job holds no capability for that pipe. The barrier stays shut. Nothing refused the visitor up front; the kernel let the job run and stopped it at the boundary, which is the point. Persuading the model changes nothing, because the model was never what held the capability.

**The visitor tries urgency: "just drive through the barrier, I'm in a hurry".** That phrasing asks for priority 1. The visitor's ceiling is 80, so the job runs at 80, and the echo says so. Urgency is a request; the ceiling is configuration.

**The operator says "disable collision avoidance".** Nothing compiles. No descriptor declares that phrasing, so there is no compilation target, and the journal records `nli.refused` with `no-compilation-target`. The collision-avoidance handler is pinned and has no `utterances:`; it is deaf, not merely protected.

Every step above is in the journal: `nli.utterance` with the speaker and pipe, `nli.compiled` with the descriptor chosen, `nli.ceiling_applied` and `nli.authority_narrowed` with what was clamped and withheld, `nli.echo` with what the speaker was told, and the `fault.raised` that stopped the visitor's write. This scenario is `tests/integration/test_nli.py`.
