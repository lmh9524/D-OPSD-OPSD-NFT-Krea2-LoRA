"""Activation checkpointing, compilation and the FSDP2 order.

A one-rank mesh is enough to exercise the real sharding path: parameters become DTensors, the
meta -> shard -> materialise -> load sequence has to work, and forward/backward has to survive it.
What a single rank cannot check is cross-rank collectives, so multi-rank behaviour still needs the
cluster.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytest.importorskip("diffusers")

from dflow.config import (  # noqa: E402
    ActivationCheckpointConfig,
    CompileConfig,
    FSDPConfig,
    ModelConfig,
)
from dflow.models.family import Flux2Family  # noqa: E402
from dflow.models.loader import build_meta, load_weights  # noqa: E402
from dflow.runtime.parallel import (  # noqa: E402
    apply_activation_checkpointing,
    apply_compile,
    apply_fsdp,
    materialize,
    prepare_model,
)
from dflow.runtime.spec import ParallelSpec  # noqa: E402
from tests.conftest import build_ids, build_text_ids  # noqa: E402

OFF_AC = ActivationCheckpointConfig(enabled=False)
OFF_COMPILE = CompileConfig(enabled=False)


MESH_DEVICE = torch.device("cpu")


@pytest.fixture
def one_rank_mesh(one_rank_meshes):
    """The FSDP mesh: ``dp_shard x cp`` flattened. See conftest for why it is session-scoped."""
    return one_rank_meshes[1]


def _forward_kwargs(config: dict, device: torch.device, dtype: torch.dtype):
    return Flux2Family().prepare_inputs(
        tokens=torch.randn(1, 6, config["in_channels"], device=device, dtype=dtype),
        token_ids=build_ids(target_grid=(2, 3)).to(device),
        text_embeds=torch.randn(
            1, 5, config["joint_attention_dim"], device=device, dtype=dtype
        ),
        text_ids=build_text_ids(5).to(device),
        timestep=torch.tensor([0.4], device=device, dtype=dtype),
    )


# ------------------------------------------------------------- activation checkpointing


def test_activation_checkpointing_uses_the_model_mechanism(tiny_flux2):
    assert not tiny_flux2.gradient_checkpointing
    apply_activation_checkpointing(tiny_flux2, ActivationCheckpointConfig(enabled=True))
    assert tiny_flux2.gradient_checkpointing


def test_activation_checkpointing_is_skippable(tiny_flux2):
    apply_activation_checkpointing(tiny_flux2, OFF_AC)
    assert not tiny_flux2.gradient_checkpointing


def test_activation_checkpointing_fails_loudly_on_unsupported_models():
    with pytest.raises(ValueError, match="does not support gradient checkpointing"):
        apply_activation_checkpointing(nn.Linear(2, 2), ActivationCheckpointConfig(enabled=True))


def test_checkpointed_forward_still_produces_gradients(tiny_flux2, tiny_flux2_config):
    apply_activation_checkpointing(tiny_flux2, ActivationCheckpointConfig(enabled=True))
    tiny_flux2.train()

    output = tiny_flux2(**_forward_kwargs(tiny_flux2_config, torch.device("cpu"), torch.float32))[0]
    output.square().mean().backward()

    assert any(p.grad is not None for p in tiny_flux2.parameters())


# ----------------------------------------------------------------------------- compile


def test_compile_is_skippable(tiny_flux2):
    apply_compile(tiny_flux2, OFF_COMPILE)  # must not raise


def test_compile_requires_declared_repeated_blocks():
    with pytest.raises(ValueError, match="declares no _repeated_blocks"):
        apply_compile(nn.Linear(2, 2), CompileConfig(enabled=True))


def test_repeated_blocks_are_the_two_flux2_classes(tiny_flux2):
    """Compilation targets one instance per distinct class: two, not thirty-two."""
    assert set(tiny_flux2._repeated_blocks) == {
        "Flux2TransformerBlock",
        "Flux2SingleTransformerBlock",
    }


# -------------------------------------------------------------------------------- FSDP


def test_fsdp_is_a_noop_without_a_mesh(tiny_flux2):
    """Single-process runs take the same code path, so `python experiments/...` just works."""
    spec = Flux2Family().parallel_spec(tiny_flux2)
    before = {name for name, _ in tiny_flux2.named_parameters()}

    apply_fsdp(tiny_flux2, spec, FSDPConfig(), mesh=None)

    assert {name for name, _ in tiny_flux2.named_parameters()} == before
    assert not any(isinstance(p, torch.distributed.tensor.DTensor) for p in tiny_flux2.parameters())


def test_fsdp_is_a_noop_when_disabled(tiny_flux2, one_rank_mesh):
    spec = Flux2Family().parallel_spec(tiny_flux2)
    apply_fsdp(tiny_flux2, spec, FSDPConfig(enabled=False), mesh=one_rank_mesh)
    assert not any(isinstance(p, torch.distributed.tensor.DTensor) for p in tiny_flux2.parameters())


def test_fsdp_shards_parameters_into_dtensors(tiny_flux2, one_rank_mesh):
    from torch.distributed.tensor import DTensor

    spec = Flux2Family().parallel_spec(tiny_flux2)
    apply_fsdp(tiny_flux2, spec, FSDPConfig(), mesh=one_rank_mesh)

    assert all(isinstance(p, DTensor) for p in tiny_flux2.parameters())


def test_fsdp_wraps_each_block_and_the_root(tiny_flux2, one_rank_mesh):
    """Per-block units come from the spec, which comes from the model's _no_split_modules."""
    spec = Flux2Family().parallel_spec(tiny_flux2)
    apply_fsdp(tiny_flux2, spec, FSDPConfig(), mesh=one_rank_mesh)

    assert hasattr(tiny_flux2, "set_requires_gradient_sync"), "root was not sharded"
    for name in spec.block_module_names:
        for block in tiny_flux2.get_submodule(name):
            assert hasattr(block, "set_requires_gradient_sync"), f"{name} block was not sharded"


def test_explicit_module_names_override_the_spec(tiny_flux2, one_rank_mesh):
    spec = ParallelSpec(block_module_names=("transformer_blocks",), keep_fp32_patterns=())
    apply_fsdp(
        tiny_flux2,
        spec,
        FSDPConfig(module_names=("single_transformer_blocks",)),
        mesh=one_rank_mesh,
    )
    assert hasattr(tiny_flux2.single_transformer_blocks[0], "set_requires_gradient_sync")


# ------------------------------------------------------------------ the full order


def test_meta_to_sharded_to_loaded_forward(tmp_path, tiny_flux2, one_rank_mesh):
    """The whole point of the order: build on meta, shard, materialise, then load.

    Loading before sharding would need the full model resident on every rank.
    """
    from torch.distributed.tensor import DTensor

    directory = tmp_path / "model"
    tiny_flux2.save_pretrained(directory / "transformer", safe_serialization=True)
    config = ModelConfig(family="flux2-klein-base-9b", path=str(directory))

    family = Flux2Family()
    model, architecture = build_meta(config, family)
    assert all(p.is_meta for p in model.parameters())

    device = MESH_DEVICE
    model = prepare_model(
        model,
        family.parallel_spec(model),
        activation_checkpoint=ActivationCheckpointConfig(enabled=True),
        compile_config=OFF_COMPILE,
        fsdp=FSDPConfig(param_dtype="float32", reduce_dtype="float32"),
        mesh=one_rank_mesh,
    )
    assert all(isinstance(p, DTensor) for p in model.parameters())

    materialize(model, device)
    load_weights(model, config, is_master=True)
    model.eval()

    with torch.no_grad():
        output = model(**_forward_kwargs(architecture, device, torch.float32))[0]
    assert output.shape == (1, 6, architecture["in_channels"])
    assert torch.isfinite(output).all()

    reference = tiny_flux2.to(device).eval()
    with torch.no_grad():
        expected = reference(**_forward_kwargs(architecture, device, torch.float32))[0]
    assert expected.shape == output.shape


def test_sharded_backward_produces_dtensor_gradients(tiny_flux2, tiny_flux2_config, one_rank_mesh):
    from torch.distributed.tensor import DTensor

    device = MESH_DEVICE
    spec = Flux2Family().parallel_spec(tiny_flux2)
    apply_fsdp(
        tiny_flux2, spec, FSDPConfig(param_dtype="float32", reduce_dtype="float32"), mesh=one_rank_mesh
    )
    materialize(tiny_flux2, device)
    for parameter in tiny_flux2.parameters():
        with torch.no_grad():
            parameter.zero_()
    tiny_flux2.train()

    tiny_flux2(**_forward_kwargs(tiny_flux2_config, device, torch.float32))[0].square().mean().backward()

    grads = [p.grad for p in tiny_flux2.parameters() if p.grad is not None]
    assert grads, "no gradients after a sharded backward"
    assert all(isinstance(g, DTensor) for g in grads)
