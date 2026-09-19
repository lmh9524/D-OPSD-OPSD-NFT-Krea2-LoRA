"""The meta-init → materialise → load path, end to end.

Round-tripping through a real ``save_pretrained`` directory exercises what the 9B run will
do, without the gated checkpoint: read config only, build on meta, materialise, then fill.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from dflow.config import ModelConfig  # noqa: E402
from dflow.models.family import Flux2Family  # noqa: E402
from dflow.models.loader import (  # noqa: E402
    build_meta,
    load_architecture_config,
    load_weights,
    read_state_dict,
    resolve_path,
)
from dflow.models.registry import REGISTRY, get_family, resolve  # noqa: E402


@pytest.fixture
def saved_model(tmp_path, tiny_flux2):
    """A diffusers-format checkpoint on disk."""
    directory = tmp_path / "model"
    tiny_flux2.save_pretrained(directory / "transformer", safe_serialization=True)
    return directory, tiny_flux2


# --------------------------------------------------------------------------- registry


def test_flux2_entries_share_one_family():
    """klein 4B/9B and dev differ in config values, not in code."""
    families = {
        id(entry.family) for entry in REGISTRY.values() if isinstance(entry.family, Flux2Family)
    }
    assert len(families) == 1
    assert isinstance(get_family("flux2-klein-base-9b"), Flux2Family)


def test_a_family_instance_is_shared_unless_a_checkpoint_flag_forces_otherwise():
    """One instance per architecture, except where the checkpoint changes behaviour.

    Krea 2 is the exception and earns it: ``is_distilled`` lives in ``model_index.json``, not in
    ``transformer/config.json``, so it cannot be read at load time the way layer counts are. Turbo
    pins ``mu = 1.15`` and Raw derives it from resolution, so the two need separate instances —
    sharing one would silently give a Turbo run the wrong noise schedule.
    """
    from dflow.models.family import Krea2Family

    krea2 = {
        name: entry.family
        for name, entry in REGISTRY.items()
        if isinstance(entry.family, Krea2Family)
    }
    assert {name: family.distilled for name, family in krea2.items()} == {
        "krea2-raw": False,
        "krea2-turbo": True,
    }
    assert len({id(family) for family in krea2.values()}) == 2


def test_klein_and_dev_stack_different_text_layers():
    assert resolve("flux2-klein-base-9b").text_out_layers == (9, 18, 27)
    assert resolve("flux2-dev").text_out_layers == (10, 20, 30)


def test_unknown_model_lists_the_known_ones():
    with pytest.raises(KeyError, match="unknown model"):
        resolve("flux3-imaginary")


def test_explicit_path_overrides_the_default_repo():
    assert resolve_path(ModelConfig(family="flux2-klein-base-9b", path="/local/x")) == "/local/x"
    assert resolve_path(ModelConfig(family="flux2-klein-base-9b")).endswith("FLUX.2-klein-base-9B")


# ------------------------------------------------------------------ config, not code


def test_architecture_comes_from_the_checkpoint(saved_model, tiny_flux2_config):
    """Nothing about the architecture is hardcoded: diffusers' defaults describe dev-32B."""
    directory, _ = saved_model
    config = load_architecture_config(
        ModelConfig(family="flux2-klein-base-9b", path=str(directory)), Flux2Family()
    )
    for key, value in tiny_flux2_config.items():
        actual = config[key]
        if isinstance(value, tuple):
            assert tuple(actual) == value, key
        else:
            assert actual == value, key
    # The class default is 8; the tiny config also uses 1 layer, so assert we did not
    # silently fall back to defaults.
    assert config["num_attention_heads"] == 2
    assert config["joint_attention_dim"] == 24


# ------------------------------------------------------------------------ meta build


def test_build_meta_allocates_nothing(saved_model):
    directory, _ = saved_model
    model, config = build_meta(
        ModelConfig(family="flux2-klein-base-9b", path=str(directory)), Flux2Family()
    )
    assert config["in_channels"] == 8
    assert all(p.is_meta for p in model.parameters()), "meta init must not allocate storage"


def test_meta_then_materialise_then_load_reproduces_weights(saved_model):
    directory, original = saved_model
    config = ModelConfig(family="flux2-klein-base-9b", path=str(directory))

    model, _ = build_meta(config, Flux2Family())
    model.to_empty(device="cpu")
    load_weights(model, config, is_master=True)

    reference = original.state_dict()
    loaded = model.state_dict()
    assert set(loaded) == set(reference)
    for key, value in reference.items():
        torch.testing.assert_close(loaded[key], value, msg=lambda m, k=key: f"{k}: {m}")


def test_loaded_model_matches_the_original_forward(saved_model, tiny_flux2_config):
    from tests.conftest import build_ids, build_text_ids

    directory, original = saved_model
    config = ModelConfig(family="flux2-klein-base-9b", path=str(directory))
    model, _ = build_meta(config, Flux2Family())
    model.to_empty(device="cpu")
    load_weights(model, config, is_master=True)
    model.eval()

    kwargs = Flux2Family().prepare_inputs(
        tokens=torch.zeros(1, 6, tiny_flux2_config["in_channels"]) + 0.1,
        token_ids=build_ids(target_grid=(2, 3)),
        text_embeds=torch.zeros(1, 5, tiny_flux2_config["joint_attention_dim"]) + 0.2,
        text_ids=build_text_ids(5),
        timestep=torch.tensor([0.25]),
    )
    with torch.no_grad():
        torch.testing.assert_close(model(**kwargs)[0], original(**kwargs)[0])


# ------------------------------------------------------------------- state-dict reader


def test_reader_handles_a_single_file(saved_model):
    directory, original = saved_model
    state = read_state_dict(directory, subfolder="transformer")
    assert set(state) == set(original.state_dict())


def test_reader_refuses_to_guess_shard_order(tmp_path, tiny_flux2):
    """Several safetensors and no index means the order is unknown — fail, do not guess."""
    from safetensors.torch import save_file

    directory = tmp_path / "sharded"
    directory.mkdir()
    save_file({"a": torch.zeros(1)}, directory / "one.safetensors")
    save_file({"b": torch.zeros(1)}, directory / "two.safetensors")

    with pytest.raises(ValueError, match="no index.json"):
        read_state_dict(directory)


def test_reader_reports_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a directory"):
        read_state_dict(tmp_path / "absent")


def test_reader_reports_an_empty_directory(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no .safetensors"):
        read_state_dict(tmp_path / "empty")


# ------------------------------------------------- LoRA base-weight loading (regression)


def _tiny_krea2():
    from dflow.vendor.krea2 import Krea2Transformer2DModel

    return Krea2Transformer2DModel(
        in_channels=16, num_layers=1, attention_head_dim=8, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=16, timestep_embed_dim=8, text_hidden_dim=16,
        num_text_layers=3, text_num_attention_heads=2, text_num_key_value_heads=1,
        text_intermediate_size=16, num_layerwise_text_blocks=1, num_refiner_text_blocks=1,
        axes_dims_rope=(4, 2, 2),
    )


def test_lora_wrapped_modules_are_found_structurally():
    """Detected by the ``base_layer`` child, not by importing a peft class."""
    from dflow.config import LoRAConfig
    from dflow.models.adapter import apply_lora
    from dflow.models.family import Krea2Family
    from dflow.models.loader import lora_wrapped_modules

    model = _tiny_krea2()
    assert lora_wrapped_modules(model) == set()
    apply_lora(model, LoRAConfig(enabled=True, rank=4), targets=Krea2Family().default_lora_targets())
    wrapped = lora_wrapped_modules(model)
    assert wrapped
    assert all(hasattr(model.get_submodule(path), "base_layer") for path in wrapped)


def test_base_weights_reach_adapter_wrapped_modules():
    """The bug this guards: LoRA-targeted projections silently kept ``to_empty()``'s zeros.

    ``add_adapter`` replaces each targeted module with a ``lora.Linear`` holding the original under
    ``.base_layer``, so the checkpoint's ``...to_q.weight`` no longer matches any parameter. With
    the ``strict=False`` that LoRA runs require, nothing raised — every attention and feed-forward
    projection stayed zero and training fine-tuned a hollowed-out backbone. Loss still fell, which
    is exactly why it went unnoticed.
    """
    import torch
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    from dflow.config import LoRAConfig
    from dflow.models.adapter import apply_lora
    from dflow.models.family import Krea2Family
    from dflow.models.loader import lora_wrapped_modules, remap_lora_base_keys

    torch.manual_seed(0)
    checkpoint = {name: value.clone() for name, value in _tiny_krea2().state_dict().items()}

    torch.manual_seed(1)
    model = _tiny_krea2()
    apply_lora(model, LoRAConfig(enabled=True, rank=4), targets=Krea2Family().default_lora_targets())
    for parameter in model.parameters():  # what to_empty() leaves behind
        parameter.data.zero_()

    wrapped = lora_wrapped_modules(model)
    set_model_state_dict(
        model,
        remap_lora_base_keys(checkpoint, wrapped),
        options=StateDictOptions(full_state_dict=True, strict=False),
    )

    parameters = dict(model.named_parameters())
    base_keys = [name for name in parameters if name.endswith("base_layer.weight")]
    assert base_keys
    for name in base_keys:
        original = name.replace(".base_layer", "")
        assert torch.equal(parameters[name], checkpoint[original]), name


def test_remap_is_a_no_op_without_adapters():
    """Full fine-tuning must not be touched by the LoRA remap."""
    import torch

    from dflow.models.loader import remap_lora_base_keys

    state_dict = {"img_in.weight": torch.zeros(2, 2)}
    assert remap_lora_base_keys(state_dict, set()) is state_dict
