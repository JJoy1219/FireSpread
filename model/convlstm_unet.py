"""ConvLSTM U-Net (Phase 5).

    Input (B, T, C, H, W) + fuel (B, H, W)
        -> fuel embedding, concatenated onto every step
        -> ConvLSTM encoder, 3 layers at [64, 128, 256], downsampling between
        -> U-Net decoder, transposed-conv upsample + skip concat
        -> 1x1 conv -> per-pixel logit

**Returns logits, not probabilities.** DESIGN.md specifies a sigmoid on the final layer,
but the loss is BCE weighted by a class-imbalance ratio that reaches ~9,400 at the median
sample. `sigmoid` followed by `BCELoss` computes `log(p)` on a saturated probability and
loses the gradient in float16; `BCEWithLogitsLoss` folds the two together with the
log-sum-exp trick and stays stable. `predict()` applies the sigmoid for inference and
metrics, so the architecture is unchanged — only where the exponential is evaluated.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.convlstm import ConvLSTMLayer


class ConvBlock(nn.Module):
    """Two 3x3 convs with GroupNorm — the decoder's workhorse.

    `dropout` uses Dropout2d (whole feature maps), not per-element dropout: adjacent
    pixels in a conv feature map are strongly correlated, so per-element dropout is a
    weak regulariser on spatial data.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConvLSTMUNet(nn.Module):
    def __init__(self, in_channels: int, fuel_classes: int, fuel_embed_dim: int = 8,
                 hidden_dims: tuple[int, ...] = (64, 128, 256), out_channels: int = 1,
                 supervise_centre: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.fuel_embed = nn.Embedding(fuel_classes, fuel_embed_dim)
        self.supervise_centre = supervise_centre
        c_in = in_channels + fuel_embed_dim

        # Encoder: a ConvLSTM per scale. Each consumes the whole sequence and emits its
        # final hidden state, which doubles as that scale's skip connection.
        dims = list(hidden_dims)
        self.encoders = nn.ModuleList()
        prev = c_in
        for d in dims:
            self.encoders.append(ConvLSTMLayer(prev, d))
            prev = d
        self.pool = nn.MaxPool2d(2)

        # Decoder: mirror back up, concatenating the skip from each encoder scale.
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(dims[i], dims[i - 1], 2, stride=2))
            self.dec.append(ConvBlock(dims[i - 1] * 2, dims[i - 1], dropout))
        self.head = nn.Conv2d(dims[0], out_channels, 1)

    def forward(self, x: torch.Tensor, fuel: torch.Tensor) -> torch.Tensor:
        """`x` is `(B, T, C, H, W)`, `fuel` is `(B, H, W)` of class indices."""
        b, t = x.shape[:2]
        # Fuel is static, so the embedding is computed once and broadcast over the
        # sequence rather than embedded T times.
        f = self.fuel_embed(fuel).permute(0, 3, 1, 2)               # (B, E, H, W)
        x = torch.cat([x, f.unsqueeze(1).expand(-1, t, -1, -1, -1)], dim=2)

        skips = []
        for i, enc in enumerate(self.encoders):
            if i > 0:
                # Pool every step of the sequence, not the aggregated state: the next
                # ConvLSTM needs a sequence, not a single frame.
                x = self.pool(x.flatten(0, 1)).unflatten(0, (b, t))
            h = enc(x)                                              # (B, D, H', W')
            skips.append(h)
            x = h.unsqueeze(1).expand(-1, t, -1, -1, -1)

        y = skips[-1]
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips[:-1])):
            y = dec(torch.cat([up(y), skip], dim=1))
        y = self.head(y)

        if self.supervise_centre:
            # Overlap-tile inference: take a wide input for context but supervise only
            # the centre, so every predicted pixel has the full receptive field of
            # surrounding terrain and upwind weather rather than a padded edge.
            m = self.supervise_centre
            o = (y.shape[-1] - m) // 2
            y = y[..., o:o + m, o:o + m]
        return y

    @torch.no_grad()
    def predict(self, x: torch.Tensor, fuel: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x, fuel))


def build_model(cfg: dict, fuel_classes: int, in_channels: int) -> ConvLSTMUNet:
    m = cfg["model"]
    return ConvLSTMUNet(
        in_channels=in_channels,
        fuel_classes=fuel_classes,
        fuel_embed_dim=int(m.get("fuel_embed_dim", 8)),
        hidden_dims=tuple(m["hidden_dims"]),
        supervise_centre=m.get("supervise_centre"),
        dropout=float(m.get("dropout", 0.0)),
    )
