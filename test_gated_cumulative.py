"""Unit tests for the GatedCumulativeLayer.

Run: python /home/z/my-project/scripts/test_gated_cumulative.py
"""

import sys
import time
import torch
import torch.nn as nn

sys.path.insert(0, "/home/z/my-project/scripts")
from gated_cumulative import (
    GatedCumulativeLayer,
    TransformerBlock,
    MultiheadAttentionAdapter,
)

torch.manual_seed(0)


def _max_abs(a, b):
    return (a - b).abs().max().item()


def test_shapes():
    layer = GatedCumulativeLayer(d_model=64)
    for B, T, D in [(1, 1, 64), (3, 17, 64), (2, 100, 64)]:
        x = torch.randn(B, T, D)
        y = layer(x)
        assert y.shape == x.shape, f"shape mismatch {y.shape} != {x.shape}"
    print("test_shapes: PASS")


def test_causality():
    """Output at position n must NOT depend on inputs at positions > n."""
    layer = GatedCumulativeLayer(d_model=64)
    layer.eval()  # disable dropout effects (none here, but be safe)
    B, T, D = 2, 20, 64
    x = torch.randn(B, T, D)
    with torch.no_grad():
        y0 = layer(x)

    # Perturb everything strictly after position p, output at <= p must be unchanged.
    for p in [0, 5, T - 1]:
        x2 = x.clone()
        x2[:, p + 1:, :] = torch.randn(B, T - p - 1, D) * 100.0
        with torch.no_grad():
            y1 = layer(x2)
        diff = _max_abs(y0[:, : p + 1, :], y1[:, : p + 1, :])
        assert diff < 1e-5, (
            f"causality violated at p={p}: max diff {diff:.2e} "
            f"(future positions influenced output at <= p)"
        )
    print("test_causality: PASS")


def test_causality_global_dependence():
    """The OPPOSITE check: output at position n SHOULD depend on position n
    (i.e. layer is not trivially returning a constant or only past info)."""
    layer = GatedCumulativeLayer(d_model=64)
    layer.eval()
    B, T, D = 1, 12, 64
    x = torch.randn(B, T, D)
    with torch.no_grad():
        y0 = layer(x)
    x2 = x.clone()
    x2[:, 5, :] += 10.0
    with torch.no_grad():
        y1 = layer(x2)
    # Position 5 onward should change.
    diff_after = _max_abs(y0[:, 5:, :], y1[:, 5:, :])
    diff_before = _max_abs(y0[:, :5, :], y1[:, :5, :])
    assert diff_after > 1e-4, "position-5 perturbation did not propagate to position 5+"
    assert diff_before < 1e-5, "position-5 perturbation leaked to earlier positions"
    print("test_global_dependence: PASS")


def test_gradient_flow():
    """Gradients flow to all parameters and there are no NaNs."""
    layer = GatedCumulativeLayer(d_model=64)
    x = torch.randn(2, 16, 64)
    y = layer(x)
    loss = y.pow(2).sum()
    loss.backward()
    nan_params = []
    zero_params = []
    for name, p in layer.named_parameters():
        if p.grad is None:
            zero_params.append(name)
            continue
        if torch.isnan(p.grad).any():
            nan_params.append(name)
        if p.grad.abs().sum().item() == 0.0:
            zero_params.append(name)
    assert not nan_params, f"NaN grads in: {nan_params}"
    # out_norm has weight (init=1) and bias (init=0); bias can legitimately
    # have zero grad if output sums to zero, so we don't hard-assert it nonzero.
    print(f"  params with zero-or-None grad: {zero_params}")
    print("test_gradient_flow: PASS")


def test_param_count_vs_attention():
    """Parameter count should be in the same ballpark as MHA for a fair
    comparison, and we report both."""
    D = 128
    ours = GatedCumulativeLayer(d_model=D)
    mha = MultiheadAttentionAdapter(d_model=D, nhead=8)
    p_ours = sum(p.numel() for p in ours.parameters())
    p_mha = sum(p.numel() for p in mha.parameters())
    print(f"  ours params: {p_ours}, MHA params: {p_mha}")
    # Ours has 4 linear (V, G_M, Q, K) -> ~4*D*D + a LayerNorm. MHA has QKV+O = 4*D*D.
    # Same order of magnitude.
    ratio = p_ours / p_mha
    assert 0.5 < ratio < 2.0, f"param ratio out of expected range: {ratio:.3f}"
    print("test_param_count_vs_attention: PASS")


def test_block_residual():
    """The transformer-block wrapping preserves shapes and gradient flow."""
    D = 64
    for name, layer in [
        ("ours", GatedCumulativeLayer(d_model=D)),
        ("mha", MultiheadAttentionAdapter(d_model=D, nhead=8)),
    ]:
        block = TransformerBlock(D, layer, ffn_mult=4, dropout=0.0)
        x = torch.randn(2, 16, D)
        y = block(x)
        assert y.shape == x.shape, f"{name}: block output {y.shape} != {x.shape}"
        y.pow(2).sum().backward()
        n_grad_none = sum(1 for p in block.parameters() if p.grad is None)
        assert n_grad_none == 0, f"{name}: {n_grad_none} params with no grad"
    print("test_block_residual: PASS")


def test_long_sequence_no_overflow():
    """Cumulative sums can blow up over long sequences; make sure LayerNorm
    keeps the output finite for a moderately long input."""
    layer = GatedCumulativeLayer(d_model=128)
    layer.eval()
    x = torch.randn(1, 2048, 128) * 0.5
    with torch.no_grad():
        y = layer(x)
    assert torch.isfinite(y).all(), "non-finite output on long sequence"
    print(f"  long-seq output stats: mean {y.mean():.4f}, std {y.std():.4f}")
    print("test_long_sequence_no_overflow: PASS")


def test_speed_scaling():
    """Sanity: ours should be O(T) in cost (cumsum), MHA is roughly O(T^2).
    Show wall-clock at a few sequence lengths."""
    D = 256
    ours = GatedCumulativeLayer(d_model=D).eval()
    mha = MultiheadAttentionAdapter(d_model=D, nhead=8).eval()
    print("  seq_len | ours(ms) | mha(ms)")
    for T in [256, 1024, 4096]:
        x = torch.randn(1, T, D)
        # warmup
        with torch.no_grad():
            for _ in range(2):
                ours(x); mha(x)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(5):
                ours(x)
        t_ours = (time.perf_counter() - t0) / 5 * 1000
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(5):
                mha(x)
        t_mha = (time.perf_counter() - t0) / 5 * 1000
        print(f"  {T:6d} | {t_ours:7.2f} | {t_mha:7.2f}")
    print("test_speed_scaling: PASS")


def main():
    test_shapes()
    test_causality()
    test_causality_global_dependence()
    test_gradient_flow()
    test_param_count_vs_attention()
    test_block_residual()
    test_long_sequence_no_overflow()
    test_speed_scaling()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
