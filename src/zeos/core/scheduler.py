# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Fixed-priority preemptive scheduling, and the suspension stack.

Three rules from core Appendix A drive everything here:

1. The highest-priority job dispatches; ties break FIFO. The candidates are the
   ``READY`` set **and the top of the suspension stack**: a job does not stop
   holding its priority because something interrupted it, and an interrupted job
   wins a tie, because it is mid-work where a ready job is between pieces of work.
2. ``RUNNING → SUSPENDED`` happens only via preemption and ``RUNNING → BLOCKED``
   only via pipe ops. **There is no yield** -- jobs cannot volunteer scheduling
   decisions, because cooperative multitasking is the disease this design treats.
3. Wakes from ``BLOCKED`` go to ``READY``, never straight to ``RUNNING``. The
   scheduler decides who runs; a wake only says who *could*.

The suspension stack is LIFO by default, which is ordinary interrupt semantics: the
most recently interrupted job resumes first as higher-priority work drains.

This module owns queue discipline and nothing else. It does not touch the machine,
segments, or the world -- so it can be reasoned about, and tested, on its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from zeos.core.ids import JobId, JobState, Priority
from zeos.core.pcb import Job

__all__ = ["Scheduler"]


class Scheduler:
    """Ready set, running job, and the suspension stack."""

    def __init__(self) -> None:
        self._jobs: dict[JobId, Job] = {}
        self._running: JobId | None = None
        #: Bottom-to-top. The last element is the most recently preempted job and
        #: therefore the next to resume.
        self._stack: list[JobId] = []

    # -- registry -----------------------------------------------------------

    def add(self, job: Job) -> None:
        self._jobs[job.job_id] = job

    def remove(self, job_id: JobId) -> None:
        self._jobs.pop(job_id, None)
        if self._running == job_id:
            self._running = None
        if job_id in self._stack:
            self._stack.remove(job_id)

    def get(self, job_id: JobId) -> Job:
        return self._jobs[job_id]

    def has(self, job_id: JobId) -> bool:
        return job_id in self._jobs

    def jobs(self) -> tuple[Job, ...]:
        """All jobs, in stable spawn order."""
        return tuple(sorted(self._jobs.values(), key=lambda j: j.seq))

    def in_state(self, state: JobState) -> tuple[Job, ...]:
        return tuple(j for j in self.jobs() if j.state is state)

    # -- running ------------------------------------------------------------

    @property
    def running(self) -> Job | None:
        return None if self._running is None else self._jobs[self._running]

    @property
    def stack(self) -> Sequence[JobId]:
        return tuple(self._stack)

    @property
    def stack_depth(self) -> int:
        return len(self._stack)

    # -- selection ----------------------------------------------------------

    def best_ready(self) -> Job | None:
        """Highest-priority READY job; ties break FIFO by spawn order.

        Sorting on ``(priority, seq)`` rather than scanning a heap keeps the choice
        obviously deterministic, which matters more here than the constant factor:
        a scheduler whose tie-breaking depended on dict ordering would produce
        journals that differ between runs for no visible reason.
        """
        candidates = [j for j in self._jobs.values() if j.state is JobState.READY]
        if not candidates:
            return None
        return min(candidates, key=lambda j: (j.current_priority, j.seq))

    def should_preempt(self) -> Job | None:
        """The READY job that should displace the running one, if any.

        Returns None when nothing outranks the running job, when the running job is
        masking interrupts (``preemptible: false``), or when nothing is ready.
        """
        contender = self.best_ready()
        if contender is None:
            return None
        running = self.running
        if running is None:
            return contender
        if not running.is_preemptible:
            return None
        return contender if contender.outranks(running) else None

    # -- transitions --------------------------------------------------------

    def dispatch(self, job: Job) -> None:
        """READY → RUNNING. The caller must have ensured nothing is running."""
        if self._running is not None:
            raise RuntimeError(f"dispatch({job.job_id}) with job {self._running} still running")
        job.state = JobState.RUNNING
        self._running = job.job_id

    def preempt(self, job: Job) -> None:
        """RUNNING → SUSPENDED, pushed onto the stack."""
        if self._running != job.job_id:
            raise RuntimeError(f"preempt({job.job_id}) but it is not running")
        job.state = JobState.SUSPENDED
        job.preempt_count += 1
        self._stack.append(job.job_id)
        self._running = None

    def block(self, job: Job) -> None:
        """RUNNING → BLOCKED. Blocking deschedules: no forward passes, and the KV
        becomes eligible for page-out."""
        if self._running != job.job_id:
            raise RuntimeError(f"block({job.job_id}) but it is not running")
        job.state = JobState.BLOCKED
        self._running = None

    def wake(self, job: Job) -> bool:
        """BLOCKED → READY. Returns False if the job was not actually blocked."""
        if job.state is not JobState.BLOCKED:
            return False
        job.state = JobState.READY
        job.blocked_on = None
        job.blocked_reason = ""
        return True

    def wake_all(self, job_ids: Iterable[JobId]) -> tuple[Job, ...]:
        woken: list[Job] = []
        for job_id in job_ids:
            job = self._jobs.get(job_id)
            if job is not None and self.wake(job):
                woken.append(job)
        return tuple(woken)

    def yield_running(self) -> None:
        """Clear the running slot without changing job state.

        Used when the running job finished or faulted; the state transition itself
        belongs to the kernel, which knows *why*.
        """
        self._running = None

    def peek_suspended(self) -> Job | None:
        """The job the stack would resume next, without unwinding it.

        Selection needs to weigh that job against the ready set before committing,
        and ``pop_suspended`` commits. Skips entries whose job is no longer
        SUSPENDED for the same reason ``pop_suspended`` does.
        """
        for job_id in reversed(self._stack):
            job = self._jobs.get(job_id)
            if job is not None and job.state is JobState.SUSPENDED:
                return job
        return None

    def pop_suspended(self) -> Job | None:
        """LIFO pop off the suspension stack. The job becomes READY, not RUNNING."""
        while self._stack:
            job_id = self._stack.pop()
            job = self._jobs.get(job_id)
            if job is None or job.state is not JobState.SUSPENDED:
                continue  # cancelled while stacked
            job.state = JobState.READY
            return job
        return None

    def drop_from_stack(self, job_id: JobId) -> bool:
        """Remove one job from the suspension stack wherever it sits.

        ``pop_suspended`` already skips jobs that died while stacked, so this is not
        needed for correctness -- but a stack holding ids of terminal jobs reports a
        wrong ``stack_depth``, and depth is journalled and asserted on.
        """
        if job_id not in self._stack:
            return False
        self._stack = [j for j in self._stack if j != job_id]
        return True

    def cancel_below(self, depth: int) -> tuple[Job, ...]:
        """Remove up to ``depth`` jobs from the top of the stack.

        Used by a handler whose emergency invalidated the work beneath it
        (core §6.3). The kernel journals each cancellation; this only unwinds.
        """
        cancelled: list[Job] = []
        for _ in range(depth):
            if not self._stack:
                break
            job = self._jobs.get(self._stack.pop())
            if job is not None:
                cancelled.append(job)
        return tuple(cancelled)

    def clear_stack(self) -> tuple[Job, ...]:
        jobs = [self._jobs[j] for j in reversed(self._stack) if j in self._jobs]
        self._stack.clear()
        return tuple(jobs)

    # -- priority inheritance ------------------------------------------------

    def inherit_priority(self, holder: Job, blocked: Job) -> bool:
        """Lend ``blocked``'s priority to ``holder`` (core §2.2).

        Prevents the classic inversion where a high-priority job waits on a
        resource held by a low-priority one that never gets scheduled. Returns
        False when the holder is already at least as urgent.
        """
        if holder.current_priority <= blocked.current_priority:
            return False
        holder.current_priority = blocked.current_priority
        holder.inherited_from = blocked.job_id
        return True

    def restore_priority(self, job: Job) -> Priority | None:
        """Drop an inherited priority back to base. Returns the restored value."""
        if job.inherited_from is None:
            return None
        job.current_priority = job.base_priority
        job.inherited_from = None
        return job.base_priority

    # -- introspection -------------------------------------------------------

    def has_runnable(self) -> bool:
        return self.running is not None or self.best_ready() is not None

    def is_quiescent(self) -> bool:
        """Nothing running, nothing ready. Everything is blocked, stacked, or done --
        which is the normal idle state, not an error."""
        return not self.has_runnable()
