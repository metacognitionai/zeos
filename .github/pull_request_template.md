<!--
PR title: describe what is the capability or bug-fix, not the implementation.
Good: "fix: replay diverges when a vector fires on the last tick"
Avoid: "fix: add null check in vectors.py"

Link context with a `Closes #<issue>` or `Related: #<issue>` line below.
-->

## What Problem This Solves

<!--
The concrete problem. For fixes: "Fixes an issue where <doing X> would
<experience Y> when <condition>." Name the affected surface: kernel, journal,
descriptor lint, CLI, docs. Do not describe the code-level cause here.
-->

## Why This Change Was Made

<!--
One or two sentences: the shipped solution, key design decisions, and
boundaries or non-goals. No file-by-file narration.
-->

## ZEOS Impact

<!--
What will this change do or what should backend implementers expect.
-->

## Evidence

<!--
Proof this works: test output, a journal excerpt, before/after
behaviour. `uv run pytest`, `uv run pytest -m determinism`, and `uv run pyright`
must all pass -- paste anything beyond that which makes validation easy.
-->
