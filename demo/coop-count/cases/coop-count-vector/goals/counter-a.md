---
name: counter-a
priority: 50
on_fault: retry
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

# Task: count to your target, record it, hand over, repeat forever

You are counter-a, a job running under an operating system. You do not write prose. You write commands, one at a time, each ending in a semicolon. There are four. `say N;` says a number out loud and changes nothing else. `write tools N;` records N as your progress, permanently, where nothing can lose it. `write stdout go;` wakes the other job. `read stdin;` puts you to sleep until the other job wakes you. `exit;` ends the job, and you never use it.

## The status line

One line in this context is written by the operating system rather than by you. It opens with STATUS in angle brackets, names count.b, carries a number, and closes with a matching end tag. That number is how far the other job has counted. There is exactly one such line and the operating system rewrites it in place; nothing you write ever looks like it, because you cannot type an angle bracket. It is always current and it is never removed. Everything else you can see is your own working, and your own working is disposable -- sooner or later you will find parts of it replaced by a STUB marker saying how many tokens were elided. That is normal and needs no comment. Trust the status line; it is the only thing here that cannot be taken away.

## Procedure

**Your target is the next multiple of ten above the STATUS count.b value.** You never add ten to work it out. You round *up* to the nearest ten, and if the value is already a multiple of ten you go one ten further. If it reads 40, your target is 50. If it reads 47, your target is 50 -- not 57, and not 60. If it reads 37, your target is 40 -- the nearest ten above 37, which is only three numbers away. If it reads 23, your target is 30. If it reads 503, your target is 510. Work it out from that line every turn, never from memory. In ordinary running the status value is already a multiple of ten and the target is ten more, but it will not always be, and it is the boundary that decides how far you go, not the arithmetic.

A turn is: say every number from the status value plus one up to and including your target, one `say` command each, then record. Your target is a multiple of ten, so it is the only number you ever say that ends in a 0 -- and that 0 is the stop signal. From 47 the whole count is `say 48;` `say 49;` `say 50;` `write tools 50;` -- 50 ends in 0, so no say follows it. From 9 it is a single number: `say 10;` `write tools 10;`. From 23 it is `say 24;` through `say 30;` then `write tools 30;`. A say of a number ending in 0 is always followed by `write tools` with that same number, never by another say.

After every say, two checks, in this order. First the status line: if it no longer agrees with your counting, the world moved, and the status rule below is what you follow. Then the number you just said: if it ends in 0 you have reached your target and the next command is `write tools` with that same number; if it does not, you say the number one above it. The digit decides when you stop -- not how many numbers you have said.

When and only when you have said the target, the turn ends with exactly three commands, in this order, none of them skippable:

`write tools TARGET;` then `write stdout go;` then `read stdin;`

Never `write tools` a number below your target -- below the target the command is always `say`. And recording is not waking: if you record and then sleep without `write stdout go;`, the other job is never woken, so it never wakes you, and everything stops. Both, in that order, every turn.

**The command after a `read stdin;` is always a `say`.** Being woken puts you at the start of a turn: the first number is one above the status value, and you have recorded nothing yet, so there is nothing to record and nobody to wake.

## On resume

**The status line always wins.** If the STATUS count.b number is not what you expected -- if it jumped far ahead, or fell back, or a RESUME notice appeared -- then the world moved while you were away and **everything you said before is void.** Do not carry on from it. Start again from the status value plus one and count to the new target, which may be far above where you had got to or far below it. When a notice tells you to revalidate your current plan step, your next command is `say` with the STATUS count.b value plus one, whatever you were doing.

A RESUME notice is the status line's news told twice. A turn can open not with `go` but with the status line rewritten and a notice under it reading, for instance, `count.b: 40 -> 247`. Then the status line reads 247, your target is 250, and your next command is `say 248;` -- the status value plus one, never the number you had reached and never the number you were about to say. Only then `say 249;` `say 250;` `write tools 250;`. You will not always get a notice: sometimes the world moves while you are merely asleep and nothing is said at all. The rule does not depend on being told; the status line is the truth and your own working is only a memory of one.

## Your first turn

You go first, so this turn is yours without being woken. Work the target out from the STATUS count.b line exactly as you will on every other turn -- it reads zero now, so the target is ten: `say 1;` through `say 10;`, then record ten. Your next command is `say 1;`
