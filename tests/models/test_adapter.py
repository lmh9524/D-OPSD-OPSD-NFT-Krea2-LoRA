"""LoRA injection.

The property under test is the one that motivates ``add_adapter`` over
``get_peft_model``: **module paths must not change**. If they do, the parallel spec, the
FSDP wrap units and every checkpoint key derived from them break.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("peft")

from dflow.config import LoRAConfig  # noqa: E402
from dflow.models.adapter import (  # noqa: E402
    apply_lora,
    freeze_base,
    parameter_summary,
    reset_lora_parameters,
)
from dflow.models.family import Flux2Family, find_block_module_names  # noqa: E402

TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.to_qkv_mlp_proj")


def test_injection_preserves_module_paths(tiny_flux2):
    before = find_block_module_names(tiny_flux2)
    before_paths = {name for name, _ in tiny_flux2.named_modules() if "lora" not in name}

    apply_lora(tiny_flux2, LoRAConfig(enabled=True, rank=4, alpha=4.0), targets=TARGETS)

    assert find_block_module_names(tiny_flux2) == before
    after_paths = {name for name, _ in tiny_flux2.named_modules() if "lora" not in name}
    # get_peft_model would have prefixed everything with base_model.model.
    assert before_paths <= after_paths
    assert not any(name.startswith("base_model.") for name, _ in tiny_flux2.named_modules())


def test_only_adapter_parameters_are_trainable(tiny_flux2):
    apply_lora(tiny_flux2, LoRAConfig(enabled=True, rank=4, alpha=4.0), targets=TARGETS)

    summary = parameter_summary(tiny_flux2)
    assert 0 < summary.trainable < summary.total
    for name, parameter in tiny_flux2.named_parameters():
        assert parameter.requires_grad == ("lora_" in name), name


def test_single_stream_blocks_are_covered(tiny_flux2):
    """to_qkv_mlp_proj is the fused projection in single-stream blocks.

    Omitting it leaves 48 of FLUX.2's 56 blocks untouched, which trains but barely learns.
    """
    apply_lora(tiny_flux2, LoRAConfig(enabled=True, rank=4, alpha=4.0), targets=TARGETS)
    adapted = {name for name, _ in tiny_flux2.named_modules() if "lora_A" in name}
    assert any("single_transformer_blocks" in name for name in adapted)
    assert any("transformer_blocks.0" in name for name in adapted)


def test_default_targets_include_the_fused_projection():
    assert "attn.to_qkv_mlp_proj" in Flux2Family().default_lora_targets()


def test_forward_still_runs_after_injection(tiny_flux2, tiny_flux2_config):
    from tests.conftest import build_ids, build_text_ids

    apply_lora(tiny_flux2, LoRAConfig(enabled=True, rank=4, alpha=4.0), targets=TARGETS)

    kwargs = Flux2Family().prepare_inputs(
        tokens=torch.randn(1, 6, tiny_flux2_config["in_channels"]),
        token_ids=build_ids(target_grid=(2, 3)),
        text_embeds=torch.randn(1, 5, tiny_flux2_config["joint_attention_dim"]),
        text_ids=build_text_ids(5),
        timestep=torch.tensor([0.4]),
    )
    output = tiny_flux2(**kwargs)[0]
    assert output.shape == (1, 6, tiny_flux2_config["in_channels"])


def test_gradients_reach_only_adapters(tiny_flux2, tiny_flux2_config):
    from tests.conftest import build_ids, build_text_ids

    apply_lora(tiny_flux2, LoRAConfig(enabled=True, rank=4, alpha=4.0), targets=TARGETS)
    kwargs = Flux2Family().prepare_inputs(
        tokens=torch.randn(1, 6, tiny_flux2_config["in_channels"]),
        token_ids=build_ids(target_grid=(2, 3)),
        text_embeds=torch.randn(1, 5, tiny_flux2_config["joint_attention_dim"]),
        text_ids=build_text_ids(5),
        timestep=torch.tensor([0.4]),
    )
    tiny_flux2(**kwargs)[0].square().mean().backward()

    with_grad = {name for name, p in tiny_flux2.named_parameters() if p.grad is not None}
    assert with_grad, "no gradients at all"
    assert all("lora_" in name for name in with_grad)


def test_reset_after_to_empty_touches_every_layer(tiny_flux2_config):
    """Adapters created on meta are uninitialised after to_empty and must be reset."""
    from dflow.vendor.flux2 import Flux2Transformer2DModel

    with torch.device("meta"):
        model = Flux2Transformer2DModel.from_config(tiny_flux2_config)
        apply_lora(model, LoRAConfig(enabled=True, rank=4, alpha=4.0), targets=TARGETS)

    model.to_empty(device="cpu")
    reset = reset_lora_parameters(model)

    assert reset > 0, "no LoRA layers reset — training would start from uninitialised memory"
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            # lora_B is zero-initialised, so the adapter starts as a no-op.
            torch.testing.assert_close(parameter, torch.zeros_like(parameter))
        if "lora_A" in name:
            assert torch.isfinite(parameter).all()


def test_disabled_config_is_rejected(tiny_flux2):
    with pytest.raises(ValueError, match="enabled=False"):
        apply_lora(tiny_flux2, LoRAConfig(enabled=False), targets=TARGETS)


def test_empty_targets_are_rejected(tiny_flux2):
    with pytest.raises(ValueError, match="at least one target"):
        apply_lora(tiny_flux2, LoRAConfig(enabled=True), targets=())


def test_freeze_base_fails_when_nothing_was_injected(tiny_flux2):
    with pytest.raises(ValueError, match="no adapter parameters found"):
        freeze_base(tiny_flux2)


def test_summary_ratio():
    from dflow.models.adapter import ParameterSummary

    summary = ParameterSummary(trainable=5, total=1000)
    assert summary.ratio == pytest.approx(0.5)
    assert "0.5000%" in summary.describe()
    assert ParameterSummary(trainable=0, total=0).ratio == 0.0
