"""Unit tests for colosseum.networks.cores."""

from __future__ import annotations

import pytest
import torch

from colosseum.networks.cores import GRUCore, LSTMCore, NoCore, WindowAttentionCore
from colosseum.networks.state import batch_size_of, cat_batch, slice_batch, tree_leaves

IN = 6


def _make(kind: str):
    if kind == "none":
        return NoCore(IN)
    if kind == "lstm":
        return LSTMCore(IN, hidden_size=10, num_layers=2)
    if kind == "gru":
        return GRUCore(IN, hidden_size=10, num_layers=1)
    if kind == "window":
        return WindowAttentionCore(IN, d_model=8, window=3, num_heads=2, num_layers=2)
    raise ValueError(kind)


KINDS = ["none", "lstm", "gru", "window"]


@pytest.mark.parametrize("kind", KINDS)
def test_output_dim_and_state_batch_dims(kind):
    core = _make(kind)
    state = core.initial_state(4)
    y, new_state = core.step(torch.randn(4, IN), state)
    assert y.shape == (4, core.output_dim)
    if kind == "none":
        assert state is None and new_state is None
    else:
        assert batch_size_of(state) == 4 and batch_size_of(new_state) == 4


@pytest.mark.parametrize("kind", KINDS)
def test_unroll_equals_step_loop_with_resets(kind):
    torch.manual_seed(0)
    core = _make(kind)
    T, B = 7, 3
    x = torch.randn(T, B, IN)
    dones = torch.zeros(T, B, dtype=torch.bool)
    dones[2, 0] = True
    dones[4, 1] = True
    dones[5, 0] = True

    state = core.initial_state(B)
    ref = []
    for t in range(T):
        y, state = core.step(x[t], state)
        ref.append(y)
        state = core.reset_state(state, dones[t])
    out = core.unroll(x, core.initial_state(B), dones)
    assert out.shape == (T, B, core.output_dim)
    assert torch.allclose(out, torch.stack(ref), atol=1e-6)


@pytest.mark.parametrize("kind", ["lstm", "gru", "window"])
def test_reset_state_replaces_only_done_rows(kind):
    torch.manual_seed(0)
    core = _make(kind)
    state = core.initial_state(3)
    for _ in range(4):
        _, state = core.step(torch.randn(3, IN), state)
    reset = core.reset_state(state, torch.tensor([False, True, False]))
    init = core.initial_state(1)
    for a, b, i in zip(tree_leaves(reset), tree_leaves(state), tree_leaves(init)):
        assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
        assert torch.equal(a[1], i[0])


@pytest.mark.parametrize("kind", ["lstm", "gru", "window"])
def test_reset_isolates_post_done_steps(kind):
    """After a done at t=1, outputs at t>=2 must not depend on inputs at t<=1."""
    torch.manual_seed(0)
    core = _make(kind)
    T, B = 5, 1
    xa = torch.randn(T, B, IN)
    xb = xa.clone()
    xb[0] = torch.randn(B, IN)
    xb[1] = torch.randn(B, IN)
    dones = torch.zeros(T, B, dtype=torch.bool)
    dones[1, 0] = True
    ya = core.unroll(xa, core.initial_state(B), dones)
    yb = core.unroll(xb, core.initial_state(B), dones)
    assert torch.allclose(ya[2:], yb[2:], atol=1e-6)
    assert not torch.allclose(ya[:2], yb[:2], atol=1e-6)


@pytest.mark.parametrize("kind", ["lstm", "gru", "window"])
def test_batched_step_equals_per_row_step(kind):
    """Grouped inference: cat_batch/slice_batch around step must not mix rows."""
    torch.manual_seed(0)
    core = _make(kind)
    rows = []
    for _ in range(3):
        s = core.initial_state(1)
        _, s = core.step(torch.randn(1, IN), s)
        rows.append(s)
    x = torch.randn(3, IN)
    y_batch, s_batch = core.step(x, cat_batch(rows))
    for i in range(3):
        y_i, s_i = core.step(x[i:i + 1], rows[i])
        assert torch.allclose(y_batch[i:i + 1], y_i, atol=1e-6)
        for a, b in zip(tree_leaves(slice_batch(s_batch, i)), tree_leaves(s_i)):
            assert torch.allclose(a.float(), b.float(), atol=1e-6)


def test_window_attention_ignores_masked_memory():
    torch.manual_seed(0)
    core = _make("window")
    x = torch.randn(2, IN)
    clean = core.initial_state(2)
    dirty = {"mem": torch.randn_like(clean["mem"]) * 100.0, "len": clean["len"].clone()}
    y_clean, _ = core.step(x, clean)
    y_dirty, _ = core.step(x, dirty)
    assert torch.allclose(y_clean, y_dirty, atol=1e-6)


def test_window_attention_sees_only_last_window_steps():
    torch.manual_seed(0)
    core = _make("window")  # window=3
    T = 6
    xa = torch.randn(T, 1, IN)
    xb = xa.clone()
    xb[0] = torch.randn(1, IN)  # differs only at t=0
    dones = torch.zeros(T, 1, dtype=torch.bool)
    ya = core.unroll(xa, core.initial_state(1), dones)
    yb = core.unroll(xb, core.initial_state(1), dones)
    assert not torch.allclose(ya[3], yb[3], atol=1e-6)  # t=3 still attends to t=0
    assert torch.allclose(ya[4:], yb[4:], atol=1e-6)  # t>=4: t=0 has left the window


def test_window_attention_len_saturates_and_reset_clears():
    core = _make("window")
    state = core.initial_state(2)
    for _ in range(5):
        _, state = core.step(torch.randn(2, IN), state)
    assert state["len"].tolist() == [3, 3]
    reset = core.reset_state(state, torch.tensor([True, False]))
    assert reset["len"].tolist() == [0, 3]
    assert torch.count_nonzero(reset["mem"][0]) == 0


@pytest.mark.parametrize("kind", ["lstm", "gru", "window"])
def test_unroll_backpropagates_into_core_parameters(kind):
    torch.manual_seed(0)
    core = _make(kind)
    out = core.unroll(torch.randn(4, 2, IN), core.initial_state(2), torch.zeros(4, 2, dtype=torch.bool))
    out.sum().backward()
    assert all(p.grad is not None for p in core.parameters())


def test_window_attention_rejects_bad_sizes():
    with pytest.raises(ValueError):
        WindowAttentionCore(IN, d_model=10, num_heads=4)
    with pytest.raises(ValueError):
        WindowAttentionCore(IN, window=0)
    with pytest.raises(ValueError, match="num_layers"):
        WindowAttentionCore(IN, num_layers=0)


@pytest.mark.parametrize("kind", ["lstm", "window"])
@pytest.mark.parametrize("empty", [{}, (), [], None])
def test_reset_state_without_tensor_leaves_returns_state_unchanged(kind, empty):
    core = _make(kind)
    assert core.reset_state(empty, torch.tensor([True, False])) is empty


def test_window_batched_step_equals_per_row_step_with_different_lengths():
    """Rows with different ``len`` (0, 1, 3 prior steps) must keep their own masks.

    Guards the head/batch ordering of the attention mask (``repeat_interleave``
    over heads, batch-major), which equal-length rows cannot detect.
    """
    torch.manual_seed(0)
    core = WindowAttentionCore(IN, d_model=8, window=4, num_heads=2, num_layers=1)
    rows = []
    for n_prior in (0, 1, 3):
        s = core.initial_state(1)
        for _ in range(n_prior):
            _, s = core.step(torch.randn(1, IN), s)
        rows.append(s)
    assert cat_batch(rows)["len"].tolist() == [0, 1, 3]
    x = torch.randn(3, IN)
    y_batch, s_batch = core.step(x, cat_batch(rows))
    for i in range(3):
        y_i, s_i = core.step(x[i:i + 1], rows[i])
        assert torch.allclose(y_batch[i:i + 1], y_i, atol=1e-6), i
        for a, b in zip(tree_leaves(slice_batch(s_batch, i)), tree_leaves(s_i)):
            assert torch.allclose(a.float(), b.float(), atol=1e-6), i


def test_window_attention_memory_rows_do_not_see_current_input():
    """Causal mask: block outputs at memory positions are independent of the current ``x``."""
    torch.manual_seed(0)
    core = WindowAttentionCore(IN, d_model=8, window=3, num_heads=2, num_layers=2)
    state = core.initial_state(2)
    for _ in range(3):  # full window: every memory row is valid
        _, state = core.step(torch.randn(2, IN), state)

    captured: list[torch.Tensor] = []
    handle = core.blocks[0].register_forward_hook(lambda mod, inp, out: captured.append(out.detach()))
    try:
        core.step(torch.randn(2, IN), state)
        core.step(torch.randn(2, IN), state)
    finally:
        handle.remove()
    out_a, out_b = captured
    assert out_a.shape == (2, 4, 8)
    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-6)  # memory rows
    assert not torch.allclose(out_a[:, -1], out_b[:, -1], atol=1e-6)  # current row does see x


@pytest.mark.parametrize("cls", [LSTMCore, GRUCore])
def test_rnn_core_matches_native_module_with_nonzero_state(cls):
    """``unroll`` and the step-loop final state equal the native ``nn.LSTM``/``nn.GRU``.

    ``num_layers=2``, batch 3 and a random non-zero ``s0`` expose ``(h, c)``
    swaps and ``reshape`` used instead of ``transpose`` between ``[B, L, H]``
    and the native ``[L, B, H]``.
    """
    torch.manual_seed(0)
    T, B, L, H = 5, 3, 2, 7
    core = cls(IN, hidden_size=H, num_layers=L)
    s0 = {k: torch.randn(B, L, H) for k in core.initial_state(B)}
    x = torch.randn(T, B, IN)
    no_dones = torch.zeros(T, B, dtype=torch.bool)

    native_s0 = {k: v.transpose(0, 1).contiguous() for k, v in s0.items()}
    with torch.no_grad():
        if cls is LSTMCore:
            y_ref, (hn, cn) = core.rnn(x, (native_s0["h"], native_s0["c"]))
            final_ref = {"h": hn.transpose(0, 1), "c": cn.transpose(0, 1)}
        else:
            y_ref, hn = core.rnn(x, native_s0["h"])
            final_ref = {"h": hn.transpose(0, 1)}

        y = core.unroll(x, s0, no_dones)
        state = s0
        for t in range(T):
            _, state = core.step(x[t], state)

    assert torch.allclose(y, y_ref, atol=1e-6)
    assert state.keys() == final_ref.keys()
    for k in final_ref:
        assert state[k].shape == (B, L, H)
        assert torch.allclose(state[k], final_ref[k], atol=1e-6), k
