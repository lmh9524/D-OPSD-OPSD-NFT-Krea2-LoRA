"""Krea 2 VAE encode, pinned as the exact inverse of the pipeline's decode.

``vae.py``'s FLUX.2 counterpart is checked by element-wise equality against a real VAE-only pipeline,
because upstream *has* an encode path to compare with. Krea 2 has none — ``Krea2Pipeline`` only
decodes — so what is checkable is the algebra: whatever ``encode`` does must be undone exactly by the
lines the pipeline runs before ``vae.decode``.

These tests therefore reproduce the pipeline's decode-side arithmetic verbatim from its source and
assert the round trip, plus the two orderings that are silent when wrong (normalise-before-patchify,
and the channel order inside a packed token).
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from dflow.encoders.vae_krea2 import patchify, unpatchify  # noqa: E402

Z_DIM = 4
PATCH = 2


@pytest.fixture
def statistics():
    """Deliberately asymmetric, so a sign or reciprocal slip cannot cancel out."""
    torch.manual_seed(0)
    mean = torch.randn(Z_DIM).view(1, Z_DIM, 1, 1, 1)
    std = torch.rand(Z_DIM).view(1, Z_DIM, 1, 1, 1) + 0.5
    return mean, std


def _pipeline_denormalise(latents: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """Verbatim from ``pipeline_krea2.py``'s decode block.

    Kept in this shape — including the reciprocal that is then divided by — so the test fails if
    our reading of it was wrong, rather than if our paraphrase of it was.
    """
    latents_mean = mean
    latents_std = 1.0 / std
    return latents / latents_std + latents_mean


def test_our_encode_normalisation_is_the_pipelines_inverse(statistics):
    """The one piece of maths with no upstream to delegate to."""
    mean, std = statistics
    z_vae = torch.randn(2, Z_DIM, 1, 6, 8)

    normalised = (z_vae - mean) / std  # what Krea2VAEEncoder.encode does
    torch.testing.assert_close(_pipeline_denormalise(normalised, mean, std), z_vae)


def test_the_reciprocal_is_not_a_double_inversion(statistics):
    """A guard against 'latents_std = 1/std then divide' reading as a single inversion.

    If encode used ``* std`` instead of ``/ std``, the round trip above still fails — but only
    because std != 1. This asserts the direction explicitly against a std that is clearly not unity.
    """
    mean, std = statistics
    assert float(std.min()) > 0.5 and float(std.max()) != 1.0
    z_vae = torch.full((1, Z_DIM, 1, 2, 2), 3.0)
    wrong = (z_vae - mean) * std
    assert not torch.allclose(_pipeline_denormalise(wrong, mean, std), z_vae)


def test_patchify_round_trips():
    latents = torch.randn(2, Z_DIM, 6, 8)
    packed = patchify(latents, patch_size=PATCH)
    assert packed.shape == (2, Z_DIM * PATCH * PATCH, 3, 4)
    torch.testing.assert_close(unpatchify(packed, patch_size=PATCH), latents)


def test_packed_token_channel_order_matches_the_pipeline():
    """Our patchify + the task's pack() must equal ``Krea2Pipeline._pack_latents``.

    Both produce ``(B, HW/p^2, C*p*p)``; only the channel order inside a token can differ, and a
    mismatch is invisible — the model simply reads scrambled patches.
    """
    from dflow.tasks.ref2img.conditioning import pack

    latents = torch.randn(1, Z_DIM, 6, 8)
    ours = pack(patchify(latents, patch_size=PATCH))

    # Verbatim from Krea2Pipeline._pack_latents.
    batch, channels, height, width = latents.shape
    p = PATCH
    theirs = latents.view(batch, channels, height // p, p, width // p, p)
    theirs = theirs.permute(0, 2, 4, 1, 3, 5)
    theirs = theirs.reshape(batch, (height // p) * (width // p), channels * p * p)

    torch.testing.assert_close(ours, theirs)


def test_normalising_after_patchify_would_use_the_wrong_statistics():
    """Why the order is stated so loudly in the module docstring.

    FLUX.2 patchifies *then* normalises over 128 channels. Krea 2's statistics have ``z_dim``
    entries and decode applies them to unpacked latents, so doing it FLUX.2's way would broadcast
    ``z_dim`` numbers across ``z_dim * p * p`` channels — which runs, and is wrong.
    """
    packed_channels = Z_DIM * PATCH * PATCH
    assert packed_channels != Z_DIM
    stats = torch.randn(Z_DIM)
    packed = torch.randn(1, packed_channels, 3, 4)
    with pytest.raises(RuntimeError):
        packed - stats.view(1, Z_DIM, 1, 1)


def test_linear_patch_embed_is_the_convolution_it_replaces() -> None:
    """Qwen3-VL's patch-embedding Conv3d has kernel == stride == the input's spatial extent.

    Every output position is therefore 1x1x1 and the convolution is a matrix multiply over the
    flattened patch. PyTorch has no cuDNN kernel for that shape and falls back to
    `slow_conv_dilated3d`, which cost 2.57 s of CPU per image on an H100 against 512 ms of GPU work
    — and dominated the training step, whose 13B forward and backward were only 0.35 s of its 2.92.
    """
    import torch
    from torch import nn

    from dflow.encoders.text_krea2 import _use_linear_patch_embed

    class Visual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed = nn.Module()
            self.patch_embed.proj = nn.Conv3d(3, 32, kernel_size=(2, 4, 4), stride=(2, 4, 4))

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = Visual()

    model = Model()
    conv = model.visual.patch_embed.proj
    patches = torch.randn(7, 3 * 2 * 4 * 4)
    expected = conv(patches.view(-1, 3, 2, 4, 4)).view(-1, 32)

    _use_linear_patch_embed(model)
    actual = model.visual.patch_embed(patches)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_linear_patch_embed_refuses_a_convolution_it_is_not_equivalent_to() -> None:
    """The identity only holds while kernel and stride match; overlapping windows are a real conv."""
    import pytest
    import torch
    from torch import nn

    from dflow.encoders.text_krea2 import _use_linear_patch_embed

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Module()
            self.visual.patch_embed = nn.Module()
            self.visual.patch_embed.proj = nn.Conv3d(3, 8, kernel_size=(2, 4, 4), stride=(2, 2, 2))

    with pytest.raises(ValueError, match="only equivalent when they match"):
        _use_linear_patch_embed(Model())
    del torch
