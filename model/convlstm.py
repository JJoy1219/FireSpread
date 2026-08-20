"""ConvLSTM cell and encoder (Phase 5).

Standard formulation from Shi et al. 2015 — the gates are a single convolution over
`[x, h]`, 3x3 with same padding, so the recurrence stays spatially local.

DESIGN.md specifies layer norm on the hidden state rather than batch norm, which is the
right call for sequences: batch statistics over a 3-step sequence with a batch of 16 are
noisy, and they leak across samples in a way that is awkward at inference. Implemented as
`GroupNorm(1, C)`, which normalises over (C, H, W) per sample — identical to LayerNorm but
without pinning the spatial dimensions into the module, so the same encoder runs at 256 or
512 px without rebuilding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel: int = 3, norm: bool = True):
        super().__init__()
        self.hidden_ch = hidden_ch
        # One conv produces all four gates; slicing is cheaper than four convolutions.
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel,
                               padding=kernel // 2, bias=not norm)
        self.norm_gates = nn.GroupNorm(4, 4 * hidden_ch) if norm else nn.Identity()
        self.norm_h = nn.GroupNorm(1, hidden_ch) if norm else nn.Identity()
        self.norm_c = nn.GroupNorm(1, hidden_ch) if norm else nn.Identity()

        # Forget-gate bias at 1.0: the standard LSTM initialisation that keeps the cell
        # state from being wiped before the recurrence has learned anything.
        if not norm:
            nn.init.zeros_(self.gates.bias)
            self.gates.bias.data[hidden_ch:2 * hidden_ch] = 1.0

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        z = self.norm_gates(self.gates(torch.cat([x, h], dim=1)))
        i, f, o, g = z.chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        c = self.norm_c(c)
        h = torch.sigmoid(o) * torch.tanh(c)
        return self.norm_h(h), c

    def init_state(self, b: int, hw: tuple[int, int], device, dtype):
        z = torch.zeros(b, self.hidden_ch, *hw, device=device, dtype=dtype)
        return z, z.clone()


class ConvLSTMLayer(nn.Module):
    """Runs a `ConvLSTMCell` over the time axis, returning the final hidden state."""

    def __init__(self, in_ch: int, hidden_ch: int, kernel: int = 3, norm: bool = True):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hidden_ch, kernel, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, T, C, H, W)` -> `(B, hidden, H, W)`."""
        b, t, _, h, w = x.shape
        state = self.cell.init_state(b, (h, w), x.device, x.dtype)
        for step in range(t):
            state = self.cell(x[:, step], state)
        return state[0]
