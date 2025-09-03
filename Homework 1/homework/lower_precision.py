from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401

from .low_precision import Linear4Bit, block_dequantize_4bit


class SmallBitNet(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, channels, group_size: int = 16):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear4Bit(channels, channels, bias=True,
                           group_size=group_size),
                torch.nn.ReLU(),
                Linear4Bit(channels, channels, bias=True,
                           group_size=group_size),
                torch.nn.ReLU(),
                Linear4Bit(channels, channels, bias=True,
                           group_size=group_size),

            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, group_size: int = 16):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> SmallBitNet:
    # TODO (extra credit): Implement a BigNet that uses in
    net = SmallBitNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
