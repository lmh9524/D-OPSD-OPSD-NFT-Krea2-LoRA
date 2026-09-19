"""Text encoding against the real 16 GB Qwen3 encoder.

Marked ``slow`` and excluded by default. Run with ``pytest -m slow``.

The cheap structural guarantees live in ``test_upstream_contract.py``; this file exists for
the one thing only real weights can establish — that the encoder's output width matches the
transformer's ``joint_attention_dim``, which is the identity that makes three-layer stacking
correct rather than merely plausible.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from dflow.config import TextEncoderConfig  # noqa: E402
from dflow.encoders.text import TextEncoder  # noqa: E402

REPO = "black-forest-labs/FLUX.2-klein-base-9B"

pytestmark = pytest.mark.slow


def _snapshot() -> pathlib.Path:
    from huggingface_hub import snapshot_download

    # Scoped to the parts we need. Asking for the whole repo raises
    # IncompleteSnapshotError, because a useful local copy deliberately omits the preview
    # images and the redundant single-file checkpoint.
    try:
        return pathlib.Path(
            snapshot_download(
                REPO,
                local_files_only=True,
                allow_patterns=["*.json", "*.jinja", "tokenizer/*", "text_encoder/*"],
            )
        )
    except Exception as error:  # noqa: BLE001 - any cache miss means "skip", not "fail"
        pytest.skip(f"{REPO} text encoder not cached locally: {type(error).__name__}: {error}")


@pytest.fixture(scope="module")
def encoder():
    root = _snapshot()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return TextEncoder.load(
        TextEncoderConfig(out_layers=(9, 18, 27), max_length=512),
        path=str(root),
        device=device,
    )


@pytest.fixture(scope="module")
def transformer_config():
    return json.loads((_snapshot() / "transformer" / "config.json").read_text())


def test_width_matches_the_transformer(encoder, transformer_config):
    """3 x 4096 == 12288 == joint_attention_dim. The whole reason out_layers has three entries."""
    assert encoder.hidden_size == 4096
    assert encoder.embed_dim == transformer_config["joint_attention_dim"]
    encoder.check_compatibility(transformer_config["joint_attention_dim"])


def test_mismatch_is_caught_at_setup(encoder):
    with pytest.raises(ValueError, match="out_layers"):
        encoder.check_compatibility(9999)


def test_encode_shapes(encoder, transformer_config):
    out = encoder.encode(["a cat holding a sign", ""])
    assert out.embeds.shape == (2, 512, transformer_config["joint_attention_dim"])
    assert out.ids.shape == (2, 512, 4)
    assert torch.isfinite(out.embeds).all()


def test_text_ids_use_the_fourth_axis_for_position(encoder):
    """(T, H, W, L): text varies only in L, leaving T free to index reference images."""
    ids = encoder.encode(["hello"]).ids
    torch.testing.assert_close(ids[0, :, :3], torch.zeros_like(ids[0, :, :3]))
    torch.testing.assert_close(
        ids[0, :, 3], torch.arange(ids.shape[1], device=ids.device, dtype=ids.dtype)
    )


def test_empty_prompt_is_valid_and_distinct(encoder):
    """klein has no guidance embedding: it uses real CFG with "" as the negative prompt.

    So the unconditional branch must produce usable embeddings, and training has to keep
    showing the model empty prompts (caption dropout) or that branch degrades.
    """
    out = encoder.encode(["a photo of a dog", ""])
    assert torch.isfinite(out.embeds[1]).all()
    assert not torch.allclose(out.embeds[0], out.embeds[1])


def test_single_prompt_is_accepted_as_a_string(encoder):
    assert encoder.encode("one prompt").embeds.shape[0] == 1


def test_empty_batch_is_rejected(encoder):
    with pytest.raises(ValueError, match="at least one prompt"):
        encoder.encode([])
