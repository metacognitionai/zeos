# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The API machine as a `MachineBackend`: the contract the kernel drives blind.

The machine takes a `JobId` as a context handle and knows nothing of rings, jobs
or priorities, so every property here is one the kernel depends on and cannot
check. `test_zeos.py` is the other half: what the kernel does with the machine.
"""

import time
from types import SimpleNamespace

import pytest
from stubs import (
    TURN,
    ScriptedClient,
    StubChatClient,
    stub_claude_machine,
    stub_machine,
)
from zeos.core.ids import JobId, TokenKind
from zeos.machine.base import Token, tokens_from_text

from zeos_space_invaders.players.zeos import (
    FORMATS,
    PARTIAL,
    OpenAIAPIMachine,
    parse_syscall,
)
from zeos_space_invaders.players.zeos.api_machine import IN, OUT, PAD, THINK, VOID

BODY = "You are a job under an operating system."


def opened(machine, descriptor="pilot", body=BODY, job=1):
    """A context with the descriptor body injected, as the kernel would leave it."""
    handle = JobId(job)
    machine.create_context(handle, descriptor)
    machine.inject(handle, tokens_from_text(body))
    return handle


def drain(machine, job, stop="read", limit=400):
    """Decode until a syscall of `stop` arrives, collecting the requests."""
    out = []
    for _ in range(limit):
        result = machine.decode(job, allow_control=False)
        if result.request.op.value != "none":
            out.append(result.request)
            if result.request.op.value == stop:
                break
    return out


# --- the syscall channel -----------------------------------------------------


@pytest.mark.parametrize(
    "clause,op,pipe,payload",
    [
        ("read stdin;", "read", "stdin", []),
        ("write stdout go now;", "write", "stdout", ["go", "now"]),
        ("exit;", "exit", None, []),
        ("say something;", "none", None, []),
    ],
)
def test_the_text_abi_reads_a_clause(clause, op, pipe, payload):
    request = parse_syscall(clause)
    assert request.op.value == op
    assert (str(request.pipe) if request.pipe else None) == pipe
    assert [t.text for t in request.payload] == payload


def test_a_clause_begins_at_a_verb_and_not_at_the_last_semicolon():
    machine = OpenAIAPIMachine(client=object(), syscall_format="text")
    job = opened(machine)
    for word in ["I'll", "wait", "for", "input.", "read", "stdin;"]:
        request = machine._request_from_text(machine._ctx[job], word)
    assert request.op.value == "read"


def test_the_schema_narrows_to_the_descriptors_own_vocabulary():
    machine = OpenAIAPIMachine(client=object(), actions=("left", "right", "shoot"))
    schema = machine.syscall_schema()["schema"]
    assert schema["properties"] == {"move": {"enum": ["left", "right", "shoot"]}}
    assert schema["required"] == ["move"]
    assert schema["additionalProperties"] is False


def test_the_schema_offers_nothing_the_model_could_get_wrong():
    machine = OpenAIAPIMachine(client=object(), actions=("left",))
    schema = machine.syscall_schema()["schema"]
    assert set(schema["properties"]) == {"move"}, "no op, no pipe, no array"
    assert "steps" not in schema["properties"]
    assert schema["properties"]["move"] != {"type": "string"}, "an enum, not free text"


def test_a_json_element_never_carries_two_calls():
    """Two requests in one decode step would be two syscalls inside one
    scheduling quantum."""
    machine = stub_machine("left")
    job = opened(machine)
    requests = drain(machine, job)
    assert [r.op.value for r in requests] == ["write", "read"]
    machine.close()


def test_whitespace_inside_a_json_string_survives():
    client = StubChatClient(["go left now"])
    machine = OpenAIAPIMachine(client=client, stall_s=0.001)
    job = opened(machine)
    write = drain(machine, job)[0]
    machine.close()
    assert [t.text for t in write.payload] == ["go", "left", "now"]


def test_an_unknown_format_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown syscall_format"):
        OpenAIAPIMachine(client=object(), syscall_format="gbnf")
    assert set(FORMATS) == {"json", "text"}


# --- what the machine refuses ------------------------------------------------


def test_control_token_decoding_is_refused_rather_than_faked():
    """ZEOS-AM §9 requires reserved-token control at the sampler, and no chat API
    exposes one, so post-filtering decoded text would fake it."""
    machine = OpenAIAPIMachine(client=object())
    job = opened(machine)
    with pytest.raises(NotImplementedError, match="sampler"):
        machine.decode(job, allow_control=True)


def test_a_decoded_token_is_never_control():
    """Kind is assigned from origin and never read off text, so a model cannot
    produce a CONTROL token however it spells itself."""
    machine = stub_machine("left")
    job = opened(machine)
    drain(machine, job)
    kinds = {
        token.kind
        for token, origin in zip(
            machine._ctx[job].tokens, machine._ctx[job].origin, strict=True
        )
        if origin == OUT
    }
    machine.close()
    assert kinds == {TokenKind.NORMAL}


def test_the_machine_reports_no_attention_and_says_so():
    """ZEOS-AM §7.2's hint channel exists so the fiction is visible in the type
    system rather than invented."""
    machine = stub_machine("left")
    job = opened(machine)
    result = machine.decode(job, allow_control=False)
    machine.close()
    assert result.attention is None
    assert result.attention_hint is not None


# --- the abstract machine's invariants ---------------------------------------


def test_blocks_are_derived_and_empty_means_zero():
    machine = OpenAIAPIMachine(client=object(), block_size=4)
    job = JobId(1)
    machine.create_context(job, "pilot")
    assert machine.stats(job).blocks == 0
    machine.inject(job, tokens_from_text("a b c d e"))
    stats = machine.stats(job)
    assert (stats.resident_tokens, stats.blocks, stats.open_segment_tokens) == (5, 2, 1)
    assert machine.blocks_for_range(job, 0, 5) == {0, 1}
    assert machine.blocks_for_range(job, 3, 3) == frozenset()


def test_no_mask_and_a_mask_of_nothing_are_different():
    """ZEOS-AM §8.2: collapsing them is a privilege escalation."""
    machine = OpenAIAPIMachine(client=object(), block_size=4)
    job = opened(machine, body="a b c d e f g h")
    assert machine.visible_blocks(job) == {0, 1}
    machine.set_mask(job, frozenset())
    assert machine.visible_blocks(job) == frozenset()


def test_an_out_of_range_mask_drops_rather_than_grants():
    """AM-I7: where the mask and the sequence disagree, visibility shrinks."""
    machine = OpenAIAPIMachine(client=object(), block_size=4)
    job = opened(machine, body="a b c d")
    machine.set_mask(job, frozenset({99}))
    assert machine.visible_blocks(job) == frozenset()


def test_a_masked_block_is_not_transmitted():
    """The mask is enforced rather than requested: content outside `visible(job)`
    is not in the request at all."""
    client = StubChatClient(["left"])
    machine = OpenAIAPIMachine(client=client, block_size=4, stall_s=0.001)
    job = opened(machine, body="secret secret secret secret visible visible")
    machine.set_mask(job, frozenset({1}))
    machine.decode(job, allow_control=False)
    sent = "\n".join(m["content"] for m in client.requests[0]["messages"])
    machine.close()
    assert "secret" not in sent and "visible" in sent


def test_a_widening_mask_does_not_cancel_the_job_own_reply():
    """The only widening that can happen mid-generation is the job's own output
    extending its open segment, so it is not a context change."""
    machine = OpenAIAPIMachine(
        client=StubChatClient(["left"], delay=0.05), block_size=4, stall_s=0.001
    )
    # Two blocks, so that narrowing to one is a narrowing at all.
    job = opened(machine, body="a b c d e f g h")
    machine.decode(job, allow_control=False)
    before = machine.cancellations
    machine.set_mask(job, frozenset({0, 1}))
    machine.set_mask(job, frozenset({0, 1, 2}))
    assert machine.cancellations == before, "a widening cancelled the job's own reply"
    machine.set_mask(job, frozenset({0}))  # a narrowing is a real context change
    assert machine.cancellations == before + 1
    machine.close()


def test_padding_is_control_and_never_reaches_the_wire():
    client = StubChatClient(["left"])
    machine = OpenAIAPIMachine(client=client, block_size=8, stall_s=0.001)
    job = opened(machine, body="a b c")
    added = machine.pad_to_block(job)
    assert added == 5
    assert machine._ctx[job].origin[-1] == PAD
    assert {t.kind for t in machine.transcript(job)[-added:]} == {TokenKind.CONTROL}
    machine.decode(job, allow_control=False)
    sent = "\n".join(m["content"] for m in client.requests[0]["messages"])
    machine.close()
    assert "<pad>" not in sent


def test_padding_does_not_invalidate_a_generation():
    """The kernel pads before almost every injection, and the request is
    byte-identical either way."""
    machine = OpenAIAPIMachine(
        client=StubChatClient(["left"]), block_size=8, stall_s=0.001
    )
    job = opened(machine, body="a b c")
    machine.decode(job, allow_control=False)
    before = machine.cancellations
    machine.pad_to_block(job)
    # Asserted before closing, which cancels everything by design.
    assert machine.cancellations == before
    machine.close()


def test_splice_reports_what_it_invalidated_and_moves_offsets():
    machine = OpenAIAPIMachine(client=object(), block_size=4)
    job = opened(machine, body="a b c d e f g h")
    result = machine.splice(job, 2, 4, tokens_from_text("X"))
    assert (result.tokens_in, result.invalidated_downstream) == (1, 4)
    assert [t.text for t in machine.transcript(job)] == list("abXefgh")


#: A reply that completes its move and then keeps talking: what a server that did
#: not enforce the schema returns, and what the `partial` policy exists for.
TAIL = '{"move": "left"} and then a good many more words follow this one'
TAIL_DELAY = 0.01


def in_flight(**kw):
    """A client still streaming `TAIL` when the first decode reaches it, because a
    stub that has already put its sentinel is a finished turn, not a cancelled one.
    """
    return ScriptedClient(TAIL, delay=TAIL_DELAY, **kw)


def mid_call(machine, job, limit=400):
    """Decode until one syscall has completed and one element sits past it, keyed
    on the transcript rather than a fixed count because a stalled decode produces
    no element.
    """
    ctx = machine._ctx[job]
    for _ in range(limit):
        machine.decode(job, allow_control=False)
        gen = ctx.gen
        if gen is not None and gen.committed < len(ctx.tokens):
            return gen
    raise AssertionError("no element ever landed past a completed call")


def test_an_upstream_splice_renumbers_the_generation_rather_than_killing_it():
    """Eviction is a change of representation, not of content."""
    machine = OpenAIAPIMachine(client=in_flight(), block_size=64, stall_s=0.002)
    job = opened(machine, body="a b c d e f g h")
    gen = mid_call(machine, job)
    before = (gen.first_offset, gen.committed)

    # Two tokens replaced by one, wholly inside the body and so wholly upstream.
    machine.splice(job, 0, 2, tokens_from_text("<stub>"))
    assert machine._ctx[job].gen is gen, "the completion was destroyed"
    assert (gen.first_offset, gen.committed) == (before[0] - 1, before[1] - 1)
    machine.close()


def test_a_splice_that_reaches_the_generations_own_output_cancels_it():
    """Told apart by position alone: the machine never has to ask which caller
    this is."""
    machine = OpenAIAPIMachine(client=in_flight(), block_size=64, stall_s=0.002)
    job = opened(machine, body="a b c d e f g h")
    gen = mid_call(machine, job)
    assert gen.first_offset < len(machine._ctx[job].tokens), "nothing decoded yet"

    machine.splice(job, gen.first_offset, len(machine._ctx[job].tokens), ())
    assert machine._ctx[job].gen is None, "a splice over its own output must cancel"
    machine.close()


def test_the_void_range_is_computed_against_post_splice_offsets():
    """ZEOS-AM §6.5 puts the obligation to renumber across a splice on whoever
    keeps offsets."""
    machine = OpenAIAPIMachine(client=in_flight(), block_size=64, stall_s=0.002)
    job = opened(machine, body="a b c d e f g h")
    gen = mid_call(machine, job)
    machine.splice(job, 0, 8, tokens_from_text("<stub 6>"))

    # Cancel for a real content change, and check the void landed on the
    # uncommitted tail rather than off the end of a shortened list.
    ctx = machine._ctx[job]
    machine.inject(job, tokens_from_text("new board"))
    assert machine.voided > 0, "the void range was computed against stale offsets"
    assert VOID in ctx.origin[gen.committed :], "voided somewhere other than the tail"


def test_trunc_at_the_end_is_a_no_op():
    machine = OpenAIAPIMachine(client=object())
    job = opened(machine, body="a b c")
    assert machine.trunc(job, 3) == 0
    assert machine.trunc(job, 1) == 2
    with pytest.raises(IndexError):
        machine.trunc(job, 99)


def test_fork_copies_the_sequence_and_inherits_the_mask():
    """A compartment child is created by forking the parent and then narrowing,
    so it must start from the parent's visibility."""
    machine = OpenAIAPIMachine(client=object(), block_size=4)
    parent = opened(machine, body="a b c d e")
    machine.set_mask(parent, frozenset({0}))
    child = JobId(2)
    assert machine.fork(parent, child) == 5
    assert machine.transcript(child) == machine.transcript(parent)
    assert machine.visible_blocks(child) == {0}


def test_every_operation_raises_without_a_context():
    machine = OpenAIAPIMachine(client=object())
    for call in (
        lambda: machine.stats(JobId(9)),
        lambda: machine.transcript(JobId(9)),
        lambda: machine.decode(JobId(9), allow_control=False),
        lambda: machine.inject(JobId(9), ()),
        lambda: machine.set_mask(JobId(9), frozenset()),
    ):
        with pytest.raises(KeyError):
            call()


# --- roles come from provenance ----------------------------------------------


def test_the_first_injected_run_is_the_system_turn():
    client = StubChatClient(["left"])
    machine = OpenAIAPIMachine(client=client, stall_s=0.001)
    job = opened(machine)
    drain(machine, job)
    machine.inject(job, tokens_from_text("a new board"))
    machine.decode(job, allow_control=False)
    roles = [m["role"] for m in client.requests[-1]["messages"]]
    machine.close()
    assert roles == ["system", "assistant", "user"]


def test_a_skipped_span_breaks_a_run_rather_than_vanishing():
    """Otherwise the descriptor body and a pipe arrival merge into one system
    message, presenting device input as the job's own instructions."""
    client = StubChatClient(["left"])
    machine = OpenAIAPIMachine(client=client, stall_s=0.001, partial="drop")
    job = opened(machine)
    for _ in range(200):
        if machine.decode(job, allow_control=False).request.op.value == "write":
            break
    machine.inject(job, tokens_from_text("a new board"))
    machine.decode(job, allow_control=False)
    messages = client.requests[-1]["messages"]
    machine.close()
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == BODY


THOUGHT = "the bomb is two rows up so I"


def test_reasoning_is_in_the_transcript_and_not_in_the_request():
    """In `T` because it is decoded work the budget sees; out of the request
    because handing a model back its own raw reasoning is the expensive way to
    resume."""
    machine = stub_claude_machine("left", thinking=THOUGHT)
    job = opened(machine)
    drain(machine, job)
    origins = machine._ctx[job].origin
    sent = (
        "\n".join(m["content"] for m in machine.client_stub.requests[-1]["messages"])
        + machine.client_stub.requests[-1]["system"][0]["text"]
    )
    machine.close()
    assert THINK in origins
    assert machine.thinking_words == len(THOUGHT.split())
    assert "two rows up" not in sent


def test_reasoning_splits_on_whitespace_whatever_the_format():
    """An element is what `stats` counts, so a paragraph of thinking arriving as
    one element would move the budget by one."""
    machine = stub_claude_machine("left", thinking="one two three four five")
    job = opened(machine)
    drain(machine, job)
    machine.close()
    assert machine.thinking_words == 5


def test_a_syscall_is_never_read_out_of_reasoning():
    """The extractor is fed only by the content channel, so a model musing about
    a write cannot be taken to have issued one."""
    machine = stub_claude_machine(
        "left", thinking='write stdout right; {"op": "write", "pipe": "stdout"}'
    )
    job = opened(machine)
    requests = drain(machine, job)
    machine.close()
    assert [r.op.value for r in requests] == ["write", "read"]
    assert [t.text for t in requests[0].payload] == ["left"]


# --- cancellation and the tail of a cancelled turn ---------------------------


def test_an_injection_invalidates_the_generation_in_flight():
    """A resume notice, a completed pipe read and a fresh sensor reading are the
    same event seen from here: the prefix moved."""
    machine = stub_machine("left", delay=0.02)
    job = opened(machine)
    machine.decode(job, allow_control=False)
    machine.decode(job, allow_control=False)
    machine.inject(job, tokens_from_text("a new board"))
    machine.close()
    assert machine.cancellations == 1


def test_cancelling_shuts_the_connection_rather_than_asking_it_to_stop():
    """A producer inside a blocking read never sees a flag, and a server that
    keeps generating makes the next generation queue behind it."""
    client = in_flight()
    machine = OpenAIAPIMachine(client=client, stall_s=0.001)
    job = opened(machine)
    mid_call(machine, job)
    stream = machine._ctx[job].gen.wire
    assert stream is not None and not stream.closed, "nothing was in flight"
    machine.inject(job, tokens_from_text("a new board"))
    assert stream.closed, "the cancelled completion kept its connection open"
    machine.close()


def test_a_cancelled_completion_leaves_no_half_clause_behind():
    """A half clause left behind is parsed together with the next completion's
    start, into a payload assembled from two replies."""
    machine = OpenAIAPIMachine(
        client=StubChatClient(["x"]), syscall_format="text", stall_s=0.001
    )
    job = opened(machine)
    machine._ctx[job].clause = "write stdout le"
    machine.inject(job, tokens_from_text("a new board"))
    assert machine._ctx[job].clause == ""
    machine.close()


def test_invalidate_drops_a_completion_the_kernel_took_the_machine_off():
    """The machine cannot see a preemption, and a resume whose diff is empty
    injects nothing, so the driver has to say so."""
    machine = stub_machine("left", delay=0.05)
    job = opened(machine)
    machine.decode(job, allow_control=False)
    assert machine._ctx[job].gen is not None
    machine.invalidate(job)
    assert machine._ctx[job].gen is None
    assert machine.cancellations == 1
    machine.invalidate(JobId(99))  # a job with no context is not an error
    machine.close()


def test_native_output_is_not_billed_to_the_model():
    machine = OpenAIAPIMachine(client=object())
    machine.register_behaviour("watcher", lambda _native: _write())
    job = JobId(8)
    machine.create_context(job, "watcher")
    machine.decode(job, allow_control=False)
    assert (machine.words, machine.native_words) == (0, 1)


def _write():
    from zeos.machine.base import DecodeResult, MachineRequest, OpKind

    return DecodeResult(
        tokens=tokens_from_text("dodging"),
        request=MachineRequest(op=OpKind.WRITE, pipe="stdout"),
    )


def test_a_native_control_token_is_refused_like_a_scripted_one():
    """`_runs` drops framing by origin rather than by kind, so a CONTROL token
    with origin OUT would put the kernel's own framing on the wire."""
    from zeos.machine.base import ControlTokenViolation, DecodeResult

    machine = OpenAIAPIMachine(client=object())
    machine.register_behaviour(
        "bad", lambda _native: DecodeResult(tokens=(Token("<pad>", TokenKind.CONTROL),))
    )
    job = JobId(9)
    machine.create_context(job, "bad")
    with pytest.raises(ControlTokenViolation):
        machine.decode(job, allow_control=False)


def test_the_partial_policy_defaults_from_the_format():
    """A half-written JSON object asked for nothing, while a half-finished
    sentence in `text` is exactly the thing worth handing back."""
    assert OpenAIAPIMachine(client=object(), syscall_format="json").partial == "syscall"
    assert OpenAIAPIMachine(client=object(), syscall_format="text").partial == "keep"
    assert set(PARTIAL) == {"syscall", "keep", "drop"}
    with pytest.raises(ValueError, match="unknown partial policy"):
        OpenAIAPIMachine(client=object(), partial="maybe")


@pytest.mark.parametrize(
    "policy,voided,tail",
    [("syscall", True, False), ("keep", False, True), ("drop", True, False)],
)
def test_the_tail_of_a_cancelled_turn_is_marked_not_deleted(policy, voided, tail):
    """The kernel has already charged, clocked, segmented and journalled these
    tokens, and `trunc` is the kernel's to call."""
    machine = OpenAIAPIMachine(client=in_flight(), stall_s=0.002, partial=policy)
    job = opened(machine)
    mid_call(machine, job)
    before = len(machine.transcript(job))
    machine.inject(job, tokens_from_text("Ada"))
    after = machine.transcript(job)
    machine.close()
    assert len(after) == before + 1, "T was shortened rather than marked"
    assert (VOID in machine._ctx[job].origin) is voided
    assert (machine._ctx[job].origin[-2] == OUT) is tail


# --- end of turn is not end of job -------------------------------------------


def test_a_turn_that_said_something_starts_another():
    """A chat API's EOS is the end of an assistant message, which happens after
    every reply, not the end of the job."""
    machine = OpenAIAPIMachine(client=StubChatClient(["left", "right"]), stall_s=0.001)
    job = opened(machine)
    drain(machine, job)  # the whole first turn, ending in a read
    exits = [
        machine.decode(job, allow_control=False).request.op.value for _ in range(400)
    ]
    machine.close()
    assert "exit" not in exits, "the job was ended by the API's turn structure"
    assert machine.generations >= 2


def test_a_turn_that_said_nothing_still_only_costs_a_turn():
    """The next turn carries a board this one never saw, so repeating is not a
    loop."""

    class Mute:
        """A server that ends the turn having said nothing on the content channel."""

        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kw: self)
            )

        def __iter__(self):
            return iter(())

        def close(self):
            pass

    machine = OpenAIAPIMachine(client=Mute(), stall_s=0.001)
    job = opened(machine)
    ops = [
        machine.decode(job, allow_control=False).request.op.value for _ in range(200)
    ]
    machine.close()
    assert "exit" not in ops, "a silent turn is not a reason to end the job"
    assert "read" in ops, "and it still ends the turn, so the next one is fresh"


def test_a_reply_is_a_turn_whatever_the_model_said():
    """A conforming move, the old three-field call and an empty object all mean
    the same thing: at most one move, then back to sleep on `stdin`."""

    def ops_for(reply, steps=40):
        client = ScriptedClient(reply)
        machine = OpenAIAPIMachine(client=client, stall_s=0.001)
        job = opened(machine)
        ops = [
            machine.decode(job, allow_control=False).request.op.value
            for _ in range(steps)
        ]
        machine.close()
        return [op for op in ops if op != "none"], len(client.requests)

    a_move, _ = ops_for(TURN.format(move="left"))
    old_form, _ = ops_for('{"steps": [{"op": "write", "pipe": "stdout"}]}')
    nothing, asked = ops_for("{}")

    assert a_move[:4] == ["write", "read", "write", "read"], "one move, then asleep"
    assert set(old_form) == {"read"}, "a reply naming no move names nothing"
    assert set(nothing) == {"read"}, "and neither does an empty object"
    assert asked < 40, "it does not spin: one request per turn, not per decode"


# --- the Claude wire format --------------------------------------------------


def test_the_body_goes_in_the_system_block_with_the_cache_breakpoint():
    """The body is the only part of a job's context that never changes -- the
    kernel pins it -- so it is where the one `cache_control` marker belongs."""
    machine = stub_claude_machine("left")
    job = opened(machine)
    machine.decode(job, allow_control=False)
    request = machine.client_stub.requests[0]
    machine.close()
    assert request["system"][0]["text"] == BODY
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_the_conversation_always_ends_with_a_user_turn():
    """ZEOS runs a job until it blocks and the chat API ends a turn after every
    reply, so the turn boundary is machine framing and has to be supplied."""
    machine = stub_claude_machine("left", "right")
    job = opened(machine)
    drain(machine, job)
    machine.inject(job, tokens_from_text("a new board"))
    machine.decode(job, allow_control=False)
    for request in machine.client_stub.requests:
        assert request["messages"], "messages may not be empty either"
        assert request["messages"][-1]["role"] == "user"
    machine.close()


def test_thinking_is_said_out_loud_in_both_directions():
    """Omitting the parameter is not turning it off, and `adaptive` with no
    `display` returns an empty thinking block with a signature."""
    off = stub_claude_machine("left")
    on = stub_claude_machine("left", thinking="a b c")
    on.thinking = True
    for machine, expected in ((off, {"type": "disabled"}), (on, None)):
        job = opened(machine)
        machine.decode(job, allow_control=False)
        sent = machine.client_stub.requests[0]["thinking"]
        machine.close()
        if expected:
            assert sent == expected
        else:
            assert sent == {"type": "adaptive", "display": "summarized"}


def test_the_claude_request_carries_the_bare_schema():
    """Anthropic's structured outputs take the schema itself; the name and the
    strict flag OpenAI wants have no counterpart."""
    machine = stub_claude_machine("left")
    job = opened(machine)
    machine.decode(job, allow_control=False)
    output = machine.client_stub.requests[0]["output_config"]
    machine.close()
    assert output["format"]["type"] == "json_schema"
    assert set(output["format"]["schema"]) == {
        "type",
        "properties",
        "required",
        "additionalProperties",
    }


def test_the_claude_effort_reaches_output_config():
    machine = stub_claude_machine("left", effort="high")
    job = opened(machine)
    machine.decode(job, allow_control=False)
    output = machine.client_stub.requests[0]["output_config"]
    machine.close()
    assert output["effort"] == "high"


def test_both_backends_report_usage_and_it_is_not_the_word_count():
    """A server reports usage only when a stream finishes, so a cancelled
    completion reports none while having cost a real prefill."""
    for machine in (stub_machine("left"), stub_claude_machine("left")):
        job = opened(machine)
        drain(machine, job)
        machine.close()
        assert machine.usage and machine.words > 0


# --- the thread the completion runs on ---------------------------------------


def test_a_cancelled_producer_is_left_to_expire_not_waited_for():
    """A producer inside time-to-first-token will not see the flag for seconds,
    and blocking would put that wait in front of the next question."""
    machine = stub_machine("left", delay=0.2)
    job = opened(machine)
    machine.decode(job, allow_control=False)
    started = time.monotonic()
    machine.inject(job, tokens_from_text("a new board"))
    assert time.monotonic() - started < 0.1, "the injection waited for the producer"
    assert machine.stranded >= 0
    machine.close(timeout=1.0)


def test_close_stops_every_producer():
    """A leaked daemon thread cannot hold the process open, but it can hold a
    connection the server is billing for."""
    machine = stub_machine("left", delay=0.01)
    job = opened(machine)
    machine.decode(job, allow_control=False)
    machine.close(timeout=1.0)
    time.sleep(0.05)
    assert machine.stranded == 0


def test_a_transport_error_is_raised_on_the_kernels_thread():
    """Raised on the producer's thread, a traceback reaches nobody."""

    class Exploding:
        @property
        def chat(self):
            raise RuntimeError("connection reset")

    machine = OpenAIAPIMachine(client=Exploding(), stall_s=0.001)
    job = opened(machine)
    with pytest.raises(RuntimeError, match="connection reset"):
        for _ in range(200):
            machine.decode(job, allow_control=False)
    machine.close()


def test_a_destroyed_context_takes_its_producer_with_it():
    machine = stub_machine("left", delay=0.05)
    job = opened(machine)
    machine.decode(job, allow_control=False)
    machine.destroy_context(job)
    assert job not in machine._ctx
    machine.close(timeout=1.0)


# --- native behaviours -------------------------------------------------------


def test_a_native_behaviour_sees_what_the_kernel_handed_it():
    """Recorded at `inject` rather than measured past a mark, because the pager
    shortens `T` and a mark can point past the end."""
    seen = []
    machine = OpenAIAPIMachine(client=object())
    machine.register_behaviour("watcher", lambda native: seen.append(native) or _exit())
    job = JobId(4)
    machine.create_context(job, "watcher")
    machine.inject(job, tokens_from_text("first"))
    machine.decode(job, allow_control=False)
    machine.inject(job, tokens_from_text("second"))
    machine.decode(job, allow_control=False)
    assert [n.arrived for n in seen] == ["first", "second"]
    assert [n.step for n in seen] == [0, 1]


def test_a_native_behaviours_arrivals_are_drained_not_repeated():
    seen = []
    machine = OpenAIAPIMachine(client=object())
    machine.register_behaviour("watcher", lambda native: seen.append(native) or _exit())
    job = JobId(5)
    machine.create_context(job, "watcher")
    machine.inject(job, tokens_from_text("once"))
    machine.decode(job, allow_control=False)
    machine.decode(job, allow_control=False)
    assert [n.arrived for n in seen] == ["once", ""]


def _exit():
    from zeos.machine.base import DecodeResult, MachineRequest, OpKind

    return DecodeResult(tokens=(), request=MachineRequest(op=OpKind.EXIT))


def test_kernel_framing_is_not_shown_to_a_native_behaviour():
    """The kernel pads to a block boundary before injecting, so a reader that
    kept the tail verbatim would see `<pad> <pad> clear: left`."""
    seen = []
    machine = OpenAIAPIMachine(client=object(), block_size=8)
    machine.register_behaviour("watcher", lambda native: seen.append(native) or _exit())
    job = JobId(6)
    machine.create_context(job, "watcher")
    machine.inject(job, [Token("<pad>", TokenKind.CONTROL), *tokens_from_text("real")])
    machine.decode(job, allow_control=False)
    assert seen[0].arrived == "real"


def test_the_transcript_records_provenance_for_every_element():
    """The machine's parallel record of who put each element in `T` is the whole
    of what rebuilds a chat request."""
    machine = stub_machine("left")
    job = opened(machine)
    drain(machine, job)
    machine.inject(job, tokens_from_text("Ada"))
    origins = machine._ctx[job].origin
    machine.close()
    assert len(origins) == len(machine.transcript(job))
    assert set(origins) <= {IN, OUT, PAD, THINK, VOID}
    assert origins[0] == IN and origins[-1] == IN
