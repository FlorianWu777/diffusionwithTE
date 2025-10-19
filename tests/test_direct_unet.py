import torch

from models.direct import DirectSpatioTemporalUNet, UNetConfig


def test_adapter_allows_extra_channels():
    config = UNetConfig(in_channels=33, out_channels=12)
    model = DirectSpatioTemporalUNet(config)
    x = torch.randn(2, 36, 12, 32, 32)
    t = torch.randint(0, 1000, (2,))

    # Should run without raising and keep the target resolution.
    out = model(x, t)
    assert out.shape == (2, config.out_channels, 12, 32, 32)


def test_adapter_is_identity_for_matching_shape():
    config = UNetConfig(in_channels=33, out_channels=12)
    model = DirectSpatioTemporalUNet(config)
    x = torch.randn(1, 33, 12, 16, 16)
    t = torch.zeros(1)

    # Run twice to ensure the cached adapter does not interfere when
    # shapes match exactly.
    first = model(x, t)
    second = model(x, t)
    assert first.shape == second.shape == (1, config.out_channels, 12, 16, 16)
