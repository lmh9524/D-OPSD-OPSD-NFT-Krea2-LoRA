"""Encoder wrappers, against real weights.

The load-bearing test here is ``test_encode_matches_the_pipeline``. ``VAEEncoder.encode``
is the one place in the codebase that *mirrors* upstream logic instead of calling it —
``Flux2KleinPipeline._encode_vae_image`` is an instance method bound to ``self.vae`` and so
cannot be reused directly. Element-wise equality against a real pipeline is what keeps that
mirror honest, because every way of getting it wrong (normalise before patchify, use config
scalars instead of BatchNorm stats, wrong posterior mode) produces plausible-looking latents
rather than an error.

Skips without the cached snapshot, so CI stays offline-clean.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

pytest.importorskip("diffusers")

from dflow.config import VAEConfig  # noqa: E402
from dflow.encoders.vae import VAEEncoder, _patchify, _unpatchify  # noqa: E402

REPO = "black-forest-labs/FLUX.2-klein-base-9B"


def _snapshot() -> pathlib.Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return pathlib.Path(
            snapshot_download(REPO, allow_patterns=["vae/*", "*.json"], local_files_only=True)
        )
    except (LocalEntryNotFoundError, OSError) as error:
        pytest.skip(f"{REPO} VAE not cached locally: {error}")


@pytest.fixture(scope="module")
def real_vae():
    from diffusers import AutoencoderKLFlux2

    return AutoencoderKLFlux2.from_pretrained(
        _snapshot(), subfolder="vae", torch_dtype=torch.float32
    ).eval()


@pytest.fixture
def encoder(real_vae):
    return VAEEncoder(
        real_vae,
        VAEConfig(encode_dtype="float32", decode_dtype="float32"),
        device=torch.device("cpu"),
    )


@pytest.fixture(scope="module")
def vae_only_pipeline(real_vae):
    """A pipeline carrying just the VAE — enough to call ``_encode_vae_image``."""
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

    return Flux2KleinPipeline(
        scheduler=None, vae=real_vae, text_encoder=None, tokenizer=None, transformer=None
    )


# ------------------------------------------------------------------ the mirror is honest


def test_encode_matches_the_pipeline(encoder, vae_only_pipeline):
    """Our encode must equal upstream's, element for element."""
    torch.manual_seed(0)
    images = torch.randn(2, 3, 64, 64).clamp(-1, 1)

    ours = encoder.encode(images, mode="mode")
    theirs = vae_only_pipeline._encode_vae_image(images, generator=None)

    assert ours.shape == theirs.shape
    torch.testing.assert_close(ours, theirs, rtol=0, atol=0)


def test_pipeline_uses_the_posterior_mode(encoder, vae_only_pipeline):
    """Upstream passes sample_mode="argmax"; equality above only holds for mode(), not sample().

    Recorded because the two differ, and which one training should use is a real decision:
    inference wants determinism, training usually wants the true posterior.
    """
    torch.manual_seed(0)
    images = torch.randn(1, 3, 64, 64).clamp(-1, 1)

    generator = torch.Generator().manual_seed(0)
    sampled = encoder.encode(images, mode="sample", generator=generator)
    moded = encoder.encode(images, mode="mode")

    assert not torch.allclose(sampled, moded), "posterior sample should differ from its mode"
    torch.testing.assert_close(
        moded, vae_only_pipeline._encode_vae_image(images, generator=None), rtol=0, atol=0
    )


# ------------------------------------------------------------------------- shape algebra


def test_effective_compression_is_sixteen(encoder):
    """8x from the conv stack, times the VAE's internal 2x2 patch."""
    assert encoder.spatial_compression == 16


def test_latent_channels_include_the_patchify_fold(encoder):
    """32 latent channels x 2x2 = 128, which is the transformer's in_channels."""
    assert encoder.latent_channels == 128


@pytest.mark.parametrize(("size", "tokens_per_side"), [(512, 32), (1024, 64)])
def test_token_counts_match_the_documented_table(encoder, size, tokens_per_side):
    """1024^2 -> 4096 tokens, 512^2 -> 1024 tokens: the numbers the cost model rests on."""
    latents = encoder.encode(torch.zeros(1, 3, size, size), mode="mode")
    assert latents.shape[-2:] == (tokens_per_side, tokens_per_side)
    assert tokens_per_side**2 == {512: 1024, 1024: 4096}[size]


# --------------------------------------------------------------------- round-trip pieces


def test_patchify_round_trip():
    torch.manual_seed(0)
    latents = torch.randn(2, 32, 8, 8)
    torch.testing.assert_close(_unpatchify(_patchify(latents)), latents)


def test_patchify_rejects_odd_dimensions():
    with pytest.raises(ValueError, match="must be even"):
        _patchify(torch.zeros(1, 32, 7, 8))


def test_normalize_round_trip(encoder):
    torch.manual_seed(0)
    latents = torch.randn(1, 128, 4, 4)
    torch.testing.assert_close(encoder.denormalize(encoder.normalize(latents)), latents, atol=1e-5, rtol=1e-5)


def test_normalization_uses_batchnorm_stats(encoder, real_vae):
    """Explicitly not config scalars: FLUX.2 keeps the statistics in a BatchNorm."""
    latents = torch.zeros(1, 128, 2, 2)
    normalized = encoder.normalize(latents)

    expected_mean = real_vae.bn.running_mean.view(1, -1, 1, 1)
    expected_std = torch.sqrt(real_vae.bn.running_var.view(1, -1, 1, 1) + real_vae.config.batch_norm_eps)
    torch.testing.assert_close(normalized, (-expected_mean / expected_std).expand_as(normalized))


def test_decode_returns_images(encoder):
    latents = encoder.encode(torch.zeros(1, 3, 64, 64), mode="mode")
    decoded = encoder.decode(latents)
    assert decoded.shape == (1, 3, 64, 64)


# -------------------------------------------------------------------------- lifecycle


def test_not_loaded_when_disabled(tmp_path):
    """Returning None keeps "latents are cached, no VAE needed" visible at the call site."""
    assert (
        VAEEncoder.load(
            VAEConfig(enabled=False), path=str(tmp_path), device=torch.device("cpu")
        )
        is None
    )


def test_encoder_is_frozen_and_eval(encoder):
    assert not encoder.model.training
    assert all(not p.requires_grad for p in encoder.model.parameters())


def test_encode_rejects_video_shaped_input(encoder):
    with pytest.raises(ValueError, match=r"expected \(B, 3, H, W\)"):
        encoder.encode(torch.zeros(1, 3, 4, 64, 64))
