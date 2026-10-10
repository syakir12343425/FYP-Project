"""
CNN Encoder-Decoder for downscaling-robust steganography.

Key differences from the Chapter 4 proposal draft, and why:

1. Payload is a FIXED 1024 bits (32x32 bit grid), not "0.5 bpp = 1.5M bits".
   The decoder's adaptive-pool design (your own Section 4.5.3) can only ever
   output a 32x32 grid = 1024 values, so the encoder must target the same
   number. This file makes that the single source of truth
   (config.message.bits) so encoder and decoder can never drift apart again.

2. Six conv layers instead of four, with BatchNorm. Four 3x3 layers give a
   9x9 receptive field, far smaller than the >=4x downscale ratio your
   pipeline must survive. More layers -> larger receptive field -> the
   network can actually spread one bit across a region large enough to
   survive resizing.

3. Residual scale defaults to 0.03 (config), not 0.1 — 0.1 allows pixel
   changes of about +-25/255, which caps PSNR near 20dB, far below your
   target of 35-38dB.
"""
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, activation="relu"):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_ch)]
    if activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif activation == "tanh":
        layers.append(nn.Tanh())
    elif activation is None:
        pass
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """
    Inputs:
        cover:   (B, 3, H, W) in [0, 1]
        message: (B, K) binary, K = message_bits (default 1024 -> 32x32)
    Output:
        stego:   (B, 3, H, W) in [0, 1]
    """

    def __init__(self, message_bits: int = 1024, base_channels: int = 64, residual_scale: float = 0.03):
        super().__init__()
        self.message_bits = message_bits
        self.message_hw = int(message_bits ** 0.5)
        assert self.message_hw ** 2 == message_bits, "message_bits must be a perfect square (e.g. 1024 = 32x32)"
        self.residual_scale = residual_scale

        c = base_channels
        # cover (3ch) + upsampled message map (1ch) = 4ch input
        self.net = nn.Sequential(
            conv_block(4, c, "relu"),
            conv_block(c, c, "relu"),
            conv_block(c, c, "relu"),
            conv_block(c, c // 2, "relu"),
            conv_block(c // 2, c // 2, "relu"),
            conv_block(c // 2, 3, "tanh"),
        )

    def _message_to_map(self, message: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Reshape (B, K) bits into a (B, 1, H, W) map by nearest-upsampling the
        (message_hw x message_hw) grid to the cover's spatial size. Nearest
        upsampling (not tiling-by-repeat) keeps each bit in one contiguous
        block, which is what the decoder's adaptive-avg-pool later expects
        to be able to average back out."""
        B = message.shape[0]
        grid = message.view(B, 1, self.message_hw, self.message_hw)
        return nn.functional.interpolate(grid, size=(H, W), mode="nearest")

    def forward(self, cover: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        B, _, H, W = cover.shape
        msg_map = self._message_to_map(message, H, W)
        x = torch.cat([cover, msg_map], dim=1)
        delta = self.net(x) * self.residual_scale
        stego = (cover + delta).clamp(0, 1)
        return stego


class Decoder(nn.Module):
    """
    Input:  downscaled stego image (B, 3, H', W') — H', W' vary (224..384)
    Output: recovered bit probabilities (B, message_bits), via sigmoid.

    Fully convolutional + AdaptiveAvgPool2d((msg_hw, msg_hw)) so it accepts
    any input resolution without architectural changes — this part of the
    original proposal design was sound and is kept.
    """

    def __init__(self, message_bits: int = 1024, base_channels: int = 64):
        super().__init__()
        self.message_bits = message_bits
        self.message_hw = int(message_bits ** 0.5)
        assert self.message_hw ** 2 == message_bits

        c = base_channels
        self.features = nn.Sequential(
            conv_block(3, c, "relu"),
            conv_block(c, c, "relu"),
            conv_block(c, c, "relu"),
            conv_block(c, c // 2, "relu"),
            conv_block(c // 2, c // 2, "relu"),
            nn.Conv2d(c // 2, 1, kernel_size=3, stride=1, padding=1),  # linear, no activation
        )
        self.pool = nn.AdaptiveAvgPool2d((self.message_hw, self.message_hw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)          # (B, 1, H', W')
        pooled = self.pool(feat)         # (B, 1, msg_hw, msg_hw)
        flat = pooled.flatten(1)         # (B, message_bits)
        return torch.sigmoid(flat)       # bit probabilities


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # quick self-test
    enc = Encoder(message_bits=1024)
    dec = Decoder(message_bits=1024)
    print(f"Encoder params: {count_parameters(enc):,}")
    print(f"Decoder params: {count_parameters(dec):,}")

    cover = torch.rand(2, 3, 512, 512)
    msg = torch.randint(0, 2, (2, 1024)).float()
    stego = enc(cover, msg)
    print("stego:", stego.shape, stego.min().item(), stego.max().item())

    downscaled = nn.functional.interpolate(stego, size=(256, 256), mode="bilinear", align_corners=False)
    recovered = dec(downscaled)
    print("recovered:", recovered.shape)

    ber = (msg != (recovered > 0.5).float()).float().mean().item()
    print(f"BER on untrained model (sanity check, expect ~0.5): {ber:.3f}")
