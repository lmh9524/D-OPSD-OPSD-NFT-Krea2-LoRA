"""The aesthetic predictor against its real weights. ``slow``: downloads ~1.7 GB.

The fast tests pin the head's architecture against shapes read from the checkpoint, which catches a
renumbered module. They cannot catch the three things that only appear once real weights and real
CLIP are involved:

* whether the published state dict actually **loads** into what we build;
* whether the embedding we feed the head is the **projected** one, at the width the head expects;
* whether the L2 normalisation is really applied, and whether the resulting scores **discriminate**
  — a head fed unnormalised embeddings still returns plausible numbers, so the only way to notice
  is to check that a structured image outscores uniform noise.

The last one is the point of this file. Everything else is a shape check with weights attached.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")

# Guarded like every other dflow-importing test file. ``dflow/__init__.py`` is the public API
# surface and re-exports from ``models/``, so importing anything under ``dflow.`` pulls in
# diffusers — even here, where nothing under test needs it. Without this the CI fast job, which
# installs no diffusers and is the gate that must stay green, errors at collection instead of
# skipping.
pytest.importorskip("diffusers")

from dflow.config.rl import AestheticRewardConfig  # noqa: E402
from dflow.rewards.aesthetic import EMBED_DIM, AestheticReward, build_head  # noqa: E402

pytestmark = pytest.mark.slow

CPU = torch.device("cpu")
SIZE = 224


@pytest.fixture(scope="module")
def reward():
    return AestheticReward.load(AestheticRewardConfig(enabled=True), device=CPU)


def _noise(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (3, SIZE, SIZE), generator=generator, dtype=torch.uint8)


def _gradient() -> torch.Tensor:
    """A smooth colour gradient — structured, unlike noise, and reliably scored higher."""
    ramp = np.linspace(0, 255, SIZE, dtype=np.uint8)
    frame = np.stack(
        [
            np.tile(ramp, (SIZE, 1)),
            np.tile(ramp[:, None], (1, SIZE)),
            np.full((SIZE, SIZE), 128, dtype=np.uint8),
        ]
    )
    return torch.from_numpy(frame)


def test_the_published_state_dict_loads_into_the_head_we_build():
    """Strict, so a missing or unexpected key fails rather than leaving a layer at its init."""
    from huggingface_hub import hf_hub_download

    config = AestheticRewardConfig(enabled=True)
    path = hf_hub_download(config.head_repo, config.head_filename)
    state = torch.load(path, map_location="cpu", weights_only=True)
    build_head().load_state_dict(state, strict=True)


def test_the_embedding_is_unit_norm_at_the_head_s_width(reward):
    """Unit norm because the head was fitted that way; 768 because that is what it consumes."""
    embeds = reward._embed(torch.stack([_noise(), _gradient()]))
    assert embeds.shape == (2, EMBED_DIM)
    torch.testing.assert_close(embeds.norm(dim=-1), torch.ones(2), rtol=1e-4, atol=1e-4)


def test_scores_land_in_the_published_range(reward):
    """Roughly 1-10. Nothing rescales it, so a value far outside means the wrong CLIP output."""
    scores = reward.score(
        torch.stack([_noise(), _gradient()]), ["a", "b"], [{}, {}]
    )
    assert scores.shape == (2,)
    assert torch.all(scores > 0.0) and torch.all(scores < 12.0), scores


def test_the_scorer_discriminates(reward):
    """The test the fast suite cannot do.

    An unnormalised embedding, or the vision tower's hidden state instead of the projection, still
    yields numbers in the right range — they just stop tracking aesthetics. A structured image
    outscoring uniform noise is the cheapest evidence that the pipeline is wired correctly.
    """
    scores = reward.score(
        torch.stack([_gradient(), _noise(1), _noise(2)]), ["a", "b", "c"], [{}, {}, {}]
    )
    assert float(scores[0]) > float(scores[1:].max()), scores


def test_scoring_is_batch_invariant(reward):
    """``batch_size`` is a memory knob and must not change a score."""
    images = torch.stack([_gradient(), _noise(1), _noise(2)])
    prompts, metadata = ["a", "b", "c"], [{}, {}, {}]

    reward.config.batch_size = 3
    together = reward.score(images, prompts, metadata)
    reward.config.batch_size = 1
    one_at_a_time = reward.score(images, prompts, metadata)
    torch.testing.assert_close(together, one_at_a_time, rtol=1e-4, atol=1e-4)


def test_a_backbone_with_the_wrong_projection_width_is_refused():
    """It loads cleanly and scores nonsense, so the check has to be explicit."""
    with pytest.raises(ValueError, match="aesthetic head"):
        AestheticReward.load(
            AestheticRewardConfig(enabled=True, clip_model="openai/clip-vit-base-patch32"),
            device=CPU,
        )
