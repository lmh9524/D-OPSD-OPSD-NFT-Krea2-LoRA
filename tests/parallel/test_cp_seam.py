"""The CP seam.

CP is not implemented. What is being tested is that the seam exists and behaves as a
no-op at cp == 1, so the call can already sit in the right place in the loop — between
``backward()`` and ``clip_grad_norm_()``. Retrofitting that call later means editing the
middle of a loop that has since grown other concerns, and getting the order wrong does
not raise.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# `dflow/__init__.py` re-exports from `models/`, so anything under `dflow.` pulls in diffusers.
pytest.importorskip("diffusers")

from dflow.config import CPConfig, DistributedConfig  # noqa: E402
from dflow.runtime.context import init_distributed  # noqa: E402
from dflow.runtime.cp import reduce_cp_gradients  # noqa: E402


def test_cp_config_defaults_to_disabled():
    config = CPConfig()
    assert config.degree == 1
    assert not config.enabled


def test_cp_config_degree_is_product():
    assert CPConfig(ulysses_degree=4).degree == 4
    assert CPConfig(ulysses_degree=2, ring_degree=2).degree == 4
    assert CPConfig(ulysses_degree=2).enabled


def test_reduce_is_a_noop_at_cp_one(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    mesh = init_distributed(DistributedConfig())

    model = nn.Linear(4, 4)
    model(torch.randn(2, 4)).sum().backward()
    before = model.weight.grad.clone()

    reduce_cp_gradients(mesh, model)

    torch.testing.assert_close(model.weight.grad, before)


def test_reduce_refuses_silently_wrong_behaviour_when_cp_enabled(monkeypatch):
    """Better an explicit failure than quietly under-scaled gradients."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    mesh = init_distributed(DistributedConfig())
    mesh_with_cp = type(mesh)(
        **{
            **{field: getattr(mesh, field) for field in mesh.__slots__},
            "cp_size": 2,
        }
    )

    with pytest.raises(NotImplementedError, match="context parallelism is not implemented"):
        reduce_cp_gradients(mesh_with_cp, nn.Linear(2, 2))
