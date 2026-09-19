"""Precomputed prompt embeddings.

The cache key is the part worth defending. It covers ``out_layers`` and ``max_length`` because
changing either produces different values for the same text — a cache built with ``(9, 18, 27)``
silently reused for ``(10, 20, 30)`` would train on the wrong conditioning with no error anywhere.
"""

from __future__ import annotations

import pytest
import torch

# Before the import below, not after: `dflow/__init__.py` re-exports from `models/`, so anything
# under `dflow.` pulls in diffusers and collection fails before a later guard can run.
pytest.importorskip("diffusers")

from dflow.encoders.cache import (  # noqa: E402
    INDEX_NAME,
    SHARD_SIZE,
    TextEmbedCache,
    prompt_key,
)

LAYERS = (9, 18, 27)


def key(prompt: str, *, layers=LAYERS, max_length: int = 512) -> str:
    return prompt_key(prompt, out_layers=layers, max_length=max_length)


# -------------------------------------------------------------------------------- keys


def test_same_prompt_same_key():
    assert key("a cat") == key("a cat")


def test_different_prompts_differ():
    assert key("a cat") != key("a dog")


def test_out_layers_are_part_of_the_key():
    """The failure this prevents is silent: same text, different embedding."""
    assert key("a cat") != key("a cat", layers=(10, 20, 30))


def test_max_length_is_part_of_the_key():
    assert key("a cat") != key("a cat", max_length=256)


def test_empty_prompt_has_a_key():
    """Caption dropout will ask for it, so it must be cacheable."""
    assert key("")


# ------------------------------------------------------------------------ round trip


def test_write_then_read(tmp_path):
    cache = TextEmbedCache(tmp_path)
    embedding = torch.randn(8, 16)
    cache.write({key("a cat"): embedding})

    torch.testing.assert_close(TextEmbedCache(tmp_path).get(key("a cat")), embedding)


def test_slices_of_one_batch_can_be_written(tmp_path):
    """The tool passes embeds[i], which share the batch tensor's storage.

    safetensors refuses aliased tensors rather than silently duplicating them, so write() must clone.
    """
    batch = torch.randn(3, 4, 8)
    cache = TextEmbedCache(tmp_path)
    cache.write({key(str(i)): batch[i] for i in range(3)})

    reopened = TextEmbedCache(tmp_path)
    for i in range(3):
        torch.testing.assert_close(reopened.get(key(str(i))), batch[i])


def test_missing_key_returns_none(tmp_path):
    assert TextEmbedCache(tmp_path).get(key("absent")) is None


def test_membership_and_length(tmp_path):
    cache = TextEmbedCache(tmp_path)
    cache.write({key("a"): torch.zeros(2, 2), key("b"): torch.zeros(2, 2)})
    assert key("a") in cache
    assert key("c") not in cache
    assert len(cache) == 2
    assert cache.exists


def test_a_fresh_directory_is_empty(tmp_path):
    assert not TextEmbedCache(tmp_path / "new").exists


# ---------------------------------------------------------------------------- sharding


def test_entries_are_sharded(tmp_path):
    cache = TextEmbedCache(tmp_path)
    cache.write({key(str(i)): torch.zeros(2) for i in range(SHARD_SIZE + 5)})

    shards = sorted(tmp_path.glob("embeds-*.safetensors"))
    assert len(shards) == 2
    assert len(cache) == SHARD_SIZE + 5


def test_appending_keeps_existing_entries(tmp_path):
    cache = TextEmbedCache(tmp_path)
    first = torch.randn(4)
    cache.write({key("a"): first})
    cache.write({key("b"): torch.randn(4)})

    reopened = TextEmbedCache(tmp_path)
    assert len(reopened) == 2
    torch.testing.assert_close(reopened.get(key("a")), first)


def test_rewriting_a_known_key_is_a_noop(tmp_path):
    """Re-running the tool should only add what is missing."""
    cache = TextEmbedCache(tmp_path)
    original = torch.ones(4)
    cache.write({key("a"): original})
    cache.write({key("a"): torch.zeros(4)})
    torch.testing.assert_close(TextEmbedCache(tmp_path).get(key("a")), original)


def test_index_is_written_last(tmp_path):
    """So a crash leaves unreferenced shards rather than an index promising missing tensors."""
    cache = TextEmbedCache(tmp_path)
    cache.write({key("a"): torch.zeros(2)})
    assert (tmp_path / INDEX_NAME).is_file()


# -------------------------------------------------------------------------- completeness


def test_require_passes_on_a_complete_cache(tmp_path):
    cache = TextEmbedCache(tmp_path)
    keys = [key("a"), key("b")]
    cache.write({k: torch.zeros(2) for k in keys})
    cache.require(keys)


def test_require_reports_misses_with_a_remedy(tmp_path):
    """Checked before training starts, because a miss after the encoder is unloaded is unrecoverable."""
    cache = TextEmbedCache(tmp_path)
    cache.write({key("a"): torch.zeros(2)})
    with pytest.raises(KeyError, match="precompute_text_embeds"):
        cache.require([key("a"), key("b")])


# --------------------------------------------------------------------- encoder wiring


def test_encoder_needs_a_model_or_a_cache():
    pytest.importorskip("diffusers")
    from dflow.config import TextEncoderConfig
    from dflow.encoders.text import TextEncoder

    with pytest.raises(ValueError, match="needs either a model or a cache"):
        TextEncoder(None, None, TextEncoderConfig(), device=torch.device("cpu"))


def test_cache_only_encoder_serves_from_the_cache(tmp_path):
    """The whole point: no model resident, so the 16 GB encoder is never loaded."""
    pytest.importorskip("diffusers")
    from dflow.config import TextEncoderConfig
    from dflow.encoders.text import TextEncoder

    config = TextEncoderConfig(out_layers=LAYERS, max_length=4)
    cache = TextEmbedCache(tmp_path)
    embedding = torch.randn(4, 12)
    cache.write({prompt_key("a cat", out_layers=LAYERS, max_length=4): embedding})

    encoder = TextEncoder(
        None, None, config, device=torch.device("cpu"), cache=cache, embed_dim=12
    )
    assert not encoder.loaded
    assert encoder.embed_dim == 12
    assert encoder.hidden_size == 4  # 12 / 3 stacked layers

    out = encoder.encode(["a cat"])
    # Returned in the encoder's configured dtype, which is what the transformer consumes.
    torch.testing.assert_close(out.embeds[0], embedding.to(out.embeds.dtype))
    assert out.ids.shape == (1, 4, 4)


def test_cache_only_encoder_refuses_an_uncached_prompt(tmp_path):
    pytest.importorskip("diffusers")
    from dflow.config import TextEncoderConfig
    from dflow.encoders.text import TextEncoder

    cache = TextEmbedCache(tmp_path)
    cache.write({prompt_key("known", out_layers=LAYERS, max_length=4): torch.zeros(4, 12)})
    encoder = TextEncoder(
        None,
        None,
        TextEncoderConfig(out_layers=LAYERS, max_length=4),
        device=torch.device("cpu"),
        cache=cache,
        embed_dim=12,
    )
    with pytest.raises(RuntimeError, match="not cached"):
        encoder.encode(["unknown"])
