import numpy as np
import pytest
import torch

import corruptions
from model import build


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    base = np.linspace(40, 220, 96, dtype=np.float32)[None, :, None]
    return np.clip(base + rng.normal(0, 20, (64, 96, 3)), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name", corruptions.CORRUPTIONS)
@pytest.mark.parametrize("severity", [1, 2, 3])
def test_corruption_shape_dtype_deterministic(img, name, severity):
    out = corruptions.apply(img, name, severity, seed=5)
    assert out.shape == img.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(out, corruptions.apply(img, name, severity, seed=5))


@pytest.mark.parametrize("name", ["gaussian_noise", "low_light", "jpeg"])
def test_corruption_severity_monotonic(img, name):
    err = [np.abs(corruptions.apply(img, name, s, seed=3).astype(float) - img).mean() for s in (1, 2, 3)]
    assert err[0] < err[1] < err[2]


@pytest.mark.parametrize("name", ["lraspp", "deeplabv3"])
def test_model_output_matches_input_resolution(name):
    torch.manual_seed(0)
    net = build(name, pretrained=False).eval()
    with torch.no_grad():
        y = net(torch.randn(1, 3, 72, 96))
    assert y.shape == (1, 21, 72, 96)
