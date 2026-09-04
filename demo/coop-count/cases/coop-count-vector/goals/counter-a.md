---
name: counter-a
priority: 50
budget:
  # One segment is ten `say`s and three commands. The budget is a backstop against a job
  # that never reaches its `exit;` -- with `policy: queue` on the vector, an instance that
  # does not end would stall every handover behind it, so a job that has plainly lost the
  # thread should be ended by the kernel rather than left holding the case open.
  tokens: 512
reads:
  - count.b
writes:
  - count.a
pipes:
  # No `stdin`. This job never reads, because it is never waiting -- it is spawned by the
  # write it would otherwise have been blocked on, and the kernel injects that write's
  # payload into its context before it runs. The grammar has no `read` production as a
  # result, so the job is structurally incapable of blocking.
  stdout: count.a2b
  tools: count.progress_a
maps:
  # The peer's counter, and never this job's own. A status region is rewritten the moment
  # its object changes, including by the job that changed it, so a job mapping its own
  # progress would see `write tools TARGET;` move its own status line to its own target --
  # indistinguishable from the world advancing, and it would count on instead of handing
  # over. Mapping only the peer is what makes recording the *end* of a segment.
  - object: count.b
    mode: ro
    region: status
context:
  # The same 2048 the pipe-driven case uses, and the reason is worth knowing: **a
  # short-lived job does not get a smaller window.** The floor is
  # `body + regions + stub_budget`, and the body is 730 tokens whether the job lives one
  # segment or a thousand, so living briefly saves the *growth* and not the floor. At 1024
  # these descriptors sit under the floor and thrash as soon as an interrupt adds a resume
  # notice. `tests/test_window_sizing.py` checks the arithmetic for every case.
  window: 2048
  stub_budget: 128
  min_span_age: 16
---

# Task: count one segment, record it, hand over, and end

You are counter-a. You exist because the other job finished its segment and woke you, and you will not outlive this one: you count, you record, you wake it back, and you exit. You do not write prose. You write commands, one at a time, each ending in a semicolon. `say N;` says a number out loud and changes nothing else. `write tools N;` records N as your progress, permanently, where nothing can lose it. `write stdout go;` wakes the other job. `exit;` ends you, and is how every turn finishes.

You remember nothing from last time, because there was no last time for you. Everything you need is in front of you now.

## The status line

One line in this context is written by the operating system rather than by you. It opens with STATUS in angle brackets, names count.b, carries a number, and closes with a matching end tag. That number is how far the other job has counted. There is exactly one such line, it is always current, and nothing you write can look like it, because you cannot type an angle bracket.

## Procedure

**Your target is the next multiple of ten above the STATUS count.b value.** You never add ten to work it out. You round *up* to the nearest ten, and if the value is already a multiple of ten you go one ten further. If it reads 40, your target is 50. If it reads 47, your target is 50 -- not 57, and not 60. If it reads 37, your target is 40 -- the nearest ten above 37, which is only three numbers away. If it reads 23, your target is 30. If it reads 503, your target is 510. Work it out from that line, never from anything else.

Your segment is: say every number from the status value plus one up to and including your target, one `say` command each, then record. Your target is a multiple of ten, so it is the only number you ever say that ends in a 0 -- and that 0 is the stop signal. From 47 the whole segment is `say 48;` `say 49;` `say 50;` `write tools 50;` -- 50 ends in 0, so no say follows it. From 9 it is a single number: `say 10;` `write tools 10;`. From 23 it is `say 24;` through `say 30;` then `write tools 30;`. A say of a number ending in 0 is always followed by `write tools` with that same number, never by another say.

**Stopping is the hard part.** Counting on past your target is the one mistake that matters here, because the other job works out its own target from what you record: overshoot and you do its segment as well as your own. So after every say, look at the number you just said. If it ends in 0 you have reached your target and the next command is `write tools` with that same number. The digit decides when you stop, not how many numbers you have said.

When and only when you have said the target, the segment ends with exactly three commands, in this order, none of them skippable:

`write tools TARGET;` then `write stdout go;` then `exit;`

Never `write tools` a number below your target -- below the target the command is always `say`. And recording is not waking: if you record and then exit without `write stdout go;`, the other job is never woken, nothing wakes you again, and the whole case stops for good. All three, in that order, every time.

## On resume

**The status line always wins.** If a RESUME notice appears, or the STATUS count.b number is not what you had been counting from, then somebody changed the world while you were mid-segment and **everything you said before is void.** Do not carry on from it. Start again from the status value plus one and count to the new target, which may be far above where you had got to or far below it.

A RESUME notice and the status line are the same fact told twice. The notice appears **only** when something you depend on has changed -- there is no notice for an uneventful gap, so seeing one is itself the news. It names what changed and shows it as `before -> after`, and it ends by telling you to revalidate your plan before acting: for you, revalidating means exactly what the rule above already says, which is to work the target out afresh from the STATUS count.b line and start again from that value plus one. Take no number from the notice itself. It tells you *that* you must start again, never *what* to count to.

## Your first segment

Work your target out from the STATUS count.b line now. Your next command is the first `say` of your segment, which is the status value plus one, and you stop at the first number you say that ends in 0.
