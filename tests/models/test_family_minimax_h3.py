"""MiniMax-H3 family adapter, checked against a real (tiny) transformer.

Same approach as ``test_family_krea2.py``: a genuinely small ``MiniMaxH3Transformer3DModel`` rather
than a mock, so the calling convention is exercised for real — the packed one-sequence layout, the
per-row modality tags and timestep map, unbatched 3-axis position ids, and two output heads whose
rows come back in the order the index tensors asked for.

These pin *mechanics*, not learned behaviour. Nothing here trains H3; what they protect is that the
vendored file still instantiates and still accepts the layout we intend to build, against the
**pinned** diffusers release — which is the real risk, because H3 does not ship in that release and
its upstream symbols could move.
"""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("diffusers")

from dflow.models.family import MiniMaxH3Family  # noqa: E402
from dflow.models.family.minimax_h3 import (  # noqa: E402
    AUDIO_SHIFT,
    DEFAULT_LORA_TARGETS,
    VIDEO_SHIFT,
)
from dflow.vendor.minimax_h3 import MiniMaxH3Transformer3DModel  # noqa: E402

HEADS = 2
HEAD_DIM = 16
HIDDEN = 32
#: The `(t, h, w)` axes share one `inv_freq` buffer, and `2 * 3 * rope_freq_dim` channels of every
#: head are rotated while the rest pass through — so `6 * rope_freq_dim <= attention_head_dim` is a
#: hard constraint on any config. The shipped one satisfies it with room to spare (96 of 128); a
#: tiny test config has to be built to respect it, which is why this is 2 and not the default 16.
ROPE_FREQ_DIM = 2
IN_CHANNELS = 4
AUDIO_IN_CHANNELS = 4
PATCH = (1, 2, 2)
TEXT_DIM = 24
#: video rows carry `in_channels * prod(patch_size)` after patchification
VIDEO_ROW_DIM = IN_CHANNELS * PATCH[0] * PATCH[1] * PATCH[2]

TAG_VIDEO, TAG_TEXT, TAG_AUDIO = 0, 1, 2


def _config() -> dict:
    return dict(
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        hidden_size=HIDDEN,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=64,
        in_channels=IN_CHANNELS,
        audio_in_channels=AUDIO_IN_CHANNELS,
        patch_size=PATCH,
        text_dim=TEXT_DIM,
        freq_dim=16,
        time_embed_hidden_dim=HIDDEN,
        time_embed_dim=16,
        rope_freq_dim=ROPE_FREQ_DIM,
        rope_theta=10000.0,
    )


def _model() -> MiniMaxH3Transformer3DModel:
    torch.manual_seed(0)
    return MiniMaxH3Transformer3DModel.from_config(_config()).eval()


def _packed(n_text: int = 3, n_video: int = 4, n_audio: int = 2):
    """One packed sequence laid out as [text | video | audio], with its index tensors."""
    seq = n_text + n_video + n_audio
    text_indices = torch.arange(n_text)
    video_indices = torch.arange(n_text, n_text + n_video)
    audio_indices = torch.arange(n_text + n_video, seq)

    token_tags = torch.empty(seq, dtype=torch.long)
    token_tags[text_indices] = TAG_TEXT
    token_tags[video_indices] = TAG_VIDEO
    token_tags[audio_indices] = TAG_AUDIO

    # two distinct noise levels: conditioning rows clean, target rows noised
    timestep = torch.tensor([0.0, 0.7])
    timestep_indices = torch.zeros(seq, dtype=torch.long)
    timestep_indices[video_indices] = 1
    timestep_indices[audio_indices] = 1

    position_ids = torch.zeros(seq, 3, dtype=torch.long)
    position_ids[:, 0] = torch.arange(seq)

    return dict(
        hidden_states=torch.randn(1, n_video, VIDEO_ROW_DIM),
        audio_hidden_states=torch.randn(1, n_audio, AUDIO_IN_CHANNELS),
        encoder_hidden_states=torch.randn(1, n_text, TEXT_DIM),
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
    )


def test_vendored_transformer_instantiates_against_the_pinned_release():
    """The whole reason H3 is vendored ahead of the pin: its imports must resolve in 0.39.0."""
    model = _model()
    assert isinstance(model, MiniMaxH3Transformer3DModel)


def test_forward_returns_both_streams_in_index_order():
    model = _model()
    packed = _packed()
    with torch.no_grad():
        out = model(**packed, return_dict=True)
    assert out.sample.shape == packed["hidden_states"].shape
    assert out.audio_sample.shape == packed["audio_hidden_states"].shape
    assert torch.isfinite(out.sample).all()
    assert torch.isfinite(out.audio_sample).all()


def test_position_ids_must_be_unbatched_three_axis():
    """`forward` raises rather than broadcasting, the same contract Krea 2 has."""
    model = _model()
    packed = _packed()
    packed["position_ids"] = packed["position_ids"].unsqueeze(0)
    with pytest.raises(ValueError, match="position_ids"):
        model(**packed)


def test_row_indexed_tensors_must_match_the_sequence_length():
    model = _model()
    packed = _packed()
    packed["token_tags"] = packed["token_tags"][:-1]
    with pytest.raises(ValueError, match="token_tags"):
        model(**packed)


def test_rotary_width_must_fit_the_head():
    """`6 * rope_freq_dim` channels are rotated per head, so a config can ask for more than exists."""
    assert 6 * ROPE_FREQ_DIM <= HEAD_DIM
    bad = _config() | {"rope_freq_dim": HEAD_DIM}  # 6x too wide
    model = MiniMaxH3Transformer3DModel.from_config(bad).eval()
    with pytest.raises(RuntimeError, match="must match the size of tensor"):
        model(**_packed())


def test_family_reports_video_layout_and_native_cp():
    family = MiniMaxH3Family()
    model = _model()
    assert family.latent_layout == "BCFHW"
    assert family.patchify_outside is True
    spec = family.parallel_spec(model)
    assert spec.sequence_dim == 1
    assert spec.has_native_cp_plan is True


def test_noise_shift_is_constant_per_modality():
    """H3 has no resolution or step-count term; both protocol arguments are inert here."""
    family = MiniMaxH3Family()
    video = family.noise_shift_mu(image_tokens=64)
    assert family.noise_shift_mu(image_tokens=65536, inference_steps=50) == video
    assert math.isclose(math.exp(video), VIDEO_SHIFT)
    assert math.isclose(math.exp(family.noise_shift_mu(image_tokens=64, audio=True)), AUDIO_SHIFT)


def test_default_lora_targets_exist_on_the_model():
    """Suffix targets are worthless if they match nothing — check against real module names."""
    model = _model()
    linear_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    for target in DEFAULT_LORA_TARGETS:
        assert any(n.endswith(target) for n in linear_names), target


def test_prepare_inputs_raises_rather_than_inventing_conditioning():
    family = MiniMaxH3Family()
    with pytest.raises(NotImplementedError, match="packed-sequence layout"):
        family.prepare_inputs(
            tokens=torch.zeros(1, 1, 1),
            token_ids=torch.zeros(1, 3),
            text_embeds=torch.zeros(1, 1, 1),
            text_ids=torch.zeros(1, 3),
            timestep=torch.zeros(1),
        )
