"""Validate the family adapter against a real klein config, not just the toy one.

Meta init costs nothing, so the real 9B architecture can be instantiated and inspected
without downloading 18 GB of weights. That makes this the test that would catch a wrong
assumption about the actual target model.

Skips when the config snapshot is absent, so CI stays offline-clean. To populate it:

    huggingface-cli download black-forest-labs/FLUX.2-klein-9B \\
        --include "*.json" "tokenizer/*" "vae/*"
"""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

pytest.importorskip("diffusers")

from dflow.models.family import Flux2Family  # noqa: E402

REPO = "black-forest-labs/FLUX.2-klein-9B"


def _snapshot() -> pathlib.Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return pathlib.Path(
            snapshot_download(REPO, allow_patterns=["*.json"], local_files_only=True)
        )
    except (LocalEntryNotFoundError, OSError) as error:
        pytest.skip(f"{REPO} config snapshot not cached locally: {error}")


@pytest.fixture(scope="module")
def klein_configs() -> dict[str, dict]:
    root = _snapshot()
    return {
        part: json.loads((root / part / "config.json").read_text())
        for part in ("transformer", "vae", "text_encoder")
    }


def test_text_encoder_layer_stacking_matches_joint_attention_dim(klein_configs):
    """The transformer's text input width is 3x the encoder's hidden size.

    That is the whole reason ``out_layers`` has three entries: FLUX.2 concatenates three
    intermediate layers per token position rather than using the final hidden state. If
    this identity ever fails, the encoder and transformer disagree and the run is garbage.
    """
    hidden = klein_configs["text_encoder"]["hidden_size"]
    joint = klein_configs["transformer"]["joint_attention_dim"]
    assert joint == 3 * hidden, f"joint_attention_dim={joint} is not 3 x hidden_size={hidden}"


def test_out_layers_are_valid_indices(klein_configs):
    """(9, 18, 27) must index into a 36-layer encoder's hidden states."""
    from dflow.models.registry import resolve

    depth = klein_configs["text_encoder"]["num_hidden_layers"]
    for layer in resolve("flux2-klein-base-9b").text_out_layers:
        # hidden_states has depth + 1 entries (embeddings first).
        assert 0 <= layer <= depth, f"layer {layer} outside 0..{depth}"


def test_latent_channels_match_transformer_input(klein_configs):
    """in_channels = latent_channels x prod(patch_size): patchify happens before the model.

    The VAE emits 32 channels; the 2x2 patchify folds them to 128, which is what
    ``x_embedder`` consumes. Getting the order wrong changes the channel count.
    """
    import math

    vae = klein_configs["vae"]
    expected = vae["latent_channels"] * math.prod(vae["patch_size"])
    assert klein_configs["transformer"]["in_channels"] == expected


def test_rope_axes_partition_the_head_dimension(klein_configs):
    """Four axes -- reference index, height, width, text position -- must sum to head_dim."""
    transformer = klein_configs["transformer"]
    assert sum(transformer["axes_dims_rope"]) == transformer["attention_head_dim"]
    assert len(transformer["axes_dims_rope"]) == 4


def test_klein_does_not_embed_guidance(klein_configs):
    """klein uses real CFG with an empty negative prompt, so guidance=None is correct.

    It also means training needs caption dropout: the model must keep seeing empty prompts
    or the unconditional branch degrades.
    """
    assert klein_configs["transformer"]["guidance_embeds"] is False
    assert not Flux2Family.requires_guidance(klein_configs["transformer"])


def test_klein_is_shallower_than_the_class_defaults(klein_configs):
    """Guards against hardcoding: diffusers' defaults describe dev-32B, not klein."""
    transformer = klein_configs["transformer"]
    assert (transformer["num_layers"], transformer["num_single_layers"]) == (8, 24)
    assert transformer["num_attention_heads"] == 32
    # The class defaults, for contrast.
    import inspect

    from dflow.vendor.flux2 import Flux2Transformer2DModel

    defaults = inspect.signature(Flux2Transformer2DModel.__init__).parameters
    assert defaults["num_single_layers"].default == 48
    assert defaults["joint_attention_dim"].default == 15360


def test_family_derives_the_spec_from_the_real_architecture(klein_configs):
    """The real model on meta: 32 blocks across two lists, native CP plan present."""
    family = Flux2Family()
    with torch.device("meta"):
        model = family.build_meta(klein_configs["transformer"])

    spec = family.parallel_spec(model)
    assert spec.block_module_names == ("transformer_blocks", "single_transformer_blocks")
    assert spec.keep_fp32_patterns == ("pos_embed", "norm")
    assert spec.has_native_cp_plan
    assert len(model.transformer_blocks) == 8
    assert len(model.single_transformer_blocks) == 24
    assert model.inner_dim == 4096
    assert all(p.is_meta for p in model.parameters())


def test_real_config_parameter_count_is_about_9b(klein_configs):
    """Sanity check that we built the model the config describes, not a default one."""
    family = Flux2Family()
    with torch.device("meta"):
        model = family.build_meta(klein_configs["transformer"])
    total = sum(p.numel() for p in model.parameters())
    assert 7e9 < total < 11e9, f"expected roughly 9B parameters, got {total / 1e9:.2f}B"
