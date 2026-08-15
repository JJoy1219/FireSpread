"""Baseline U-Net (Phase 5).

The ablation that says whether the ConvLSTM earns its cost. Identical encoder widths,
decoder and head to `ConvLSTMUNet`, but the time axis is folded into channels instead of
being modelled recurrently.

**Not parameter-matched**: 1.90 M against the ConvLSTM's 5.35 M, because a ConvLSTM layer
carries four gate convolutions where a ConvBlock carries two plain ones. Depth and width
are matched, capacity is not, so a win for the ConvLSTM is not automatically a win for
*recurrence* — it could be a win for parameters. If the gap is small, widen this baseline
to ~5.3 M before drawing a conclusion.

Returns logits, for the same numerical reason as the ConvLSTM version.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.convlstm_unet import ConvBlock


class UNet(nn.Module):
    def __init__(self, in_channels: int, t_steps: int, fuel_classes: int,
                 fuel_embed_dim: int = 8, hidden_dims: tuple[int, ...] = (64, 128, 256),
                 out_channels: int = 1, supervise_centre: int | None = None):
        super().__init__()
        self.fuel_embed = nn.Embedding(fuel_classes, fuel_embed_dim)
        self.supervise_centre = supervise_centre
        dims = list(hidden_dims)
        c_in = in_channels * t_steps + fuel_embed_dim

        self.encoders = nn.ModuleList()
        prev = c_in
        for d in dims:
            self.encoders.append(ConvBlock(prev, d))
            prev = d
        self.pool = nn.MaxPool2d(2)

        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(dims[i], dims[i - 1], 2, stride=2))
            self.dec.append(ConvBlock(dims[i - 1] * 2, dims[i - 1]))
        self.head = nn.Conv2d(dims[0], out_channels, 1)

    def forward(self, x: torch.Tensor, fuel: torch.Tensor) -> torch.Tensor:
        f = self.fuel_embed(fuel).permute(0, 3, 1, 2)
        x = torch.cat([x.flatten(1, 2), f], dim=1)        # (B, T*C + E, H, W)

        skips = []
        for i, enc in enumerate(self.encoders):
            if i > 0:
                x = self.pool(x)
            x = enc(x)
            skips.append(x)

        y = skips[-1]
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips[:-1])):
            y = dec(torch.cat([up(y), skip], dim=1))
        y = self.head(y)

        if self.supervise_centre:
            m = self.supervise_centre
            o = (y.shape[-1] - m) // 2
            y = y[..., o:o + m, o:o + m]
        return y

    @torch.no_grad()
    def predict(self, x: torch.Tensor, fuel: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x, fuel))
