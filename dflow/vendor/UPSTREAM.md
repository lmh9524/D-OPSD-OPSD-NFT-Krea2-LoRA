# Vendored upstream code

Model definitions copied from `diffusers` so we can modify them. Everything else
(mixins, attention dispatch, embeddings, normalization, CP plumbing) is **imported
from the installed `diffusers` package** — do not vendor infrastructure.

## Pinned upstream

| | |
|---|---|
| package | `diffusers` |
| version | **`0.39.0`** (PyPI release; `pip install diffusers==0.39.0`) |
| tag | `v0.39.0` |
| pinned in | `pyproject.toml` |

Because we import private members — pipeline static methods,
`diffusers.models._modeling_parallel`, `diffusers.hooks.context_parallel` — the version **must** stay
pinned. Private paths can move without a deprecation cycle, and the failure mode is silent: different
conditioning, not an ImportError.

The vendored file was originally copied from `0.40.0.dev0 @ 6f2010e8b`. Moving the pin back to the
`v0.39.0` release was verified rather than assumed:

* `git diff v0.39.0 6f2010e8b` is **empty** for every file we depend on — `transformer_flux2.py`,
  `pipeline_flux2{,_klein}.py`, `autoencoder_kl_flux2.py`, `hooks/context_parallel.py`,
  `models/_modeling_parallel.py`;
* `Flux2LoraLoaderMixin`, `enable_parallelism`, `compile_repeated_blocks` and
  `enable_gradient_checkpointing` are byte-identical;
* the only touched-file difference that matters is five lines in
  `scheduling_flow_match_euler_discrete.set_timesteps`, on the inference path. Training uses the
  scheduler's `time_shift` only, which is unchanged and asserted element-wise in
  `tests/test_flow_matching.py`;
* the full suite passes against the PyPI build — 258 fast, 7 slow against real weights — and a 9B
  training run reproduces its losses to the digit (27.5489, 58.8703).

  **Those two loss values are not a healthy baseline.** They were recorded before the LoRA
  base-weight loading bug was found (see `models/loader.py::remap_lora_base_keys`): every
  LoRA-targeted projection was zero, so the run was reproducible but not meaningful. They remain
  useful only as a *bit-exactness* check between two builds, not as evidence the model trains.
  Re-record them against a fixed run before relying on them again.

## Files

| vendored | upstream path | lines |
|---|---|---|
| `flux2/transformer_flux2.py` | `src/diffusers/models/transformers/transformer_flux2.py` | 1386 |
| `krea2/transformer_krea2.py` | `src/diffusers/models/transformers/transformer_krea2.py` | 522 |

Not vendored on purpose:

- `autoencoder_kl_flux2.py` — VAE is frozen and never trained. Vendoring it would
  drag in `models/autoencoders/vae.py` (927 lines) and several attention processors.
  Copy it only if we need to change encode/decode or tiling.
- `pipeline_flux2_klein.py` — training does not need a pipeline. The four static
  helpers we do need (`_prepare_latent_ids`, `_prepare_image_ids`,
  `_patchify_latents`, `_pack_latents`) are copied into
  `dflow/tasks/ref2img/conditioning.py` with a provenance comment, because they are
  the only guarantee that training and inference build identical position ids.

## Local modifications

### `flux2/transformer_flux2.py`

1. **Import rewrite** (mechanical, no behaviour change) — relative imports were
   repointed at the installed package:

   ```
   from ...X   ->  from diffusers.X
   from ..X    ->  from diffusers.models.X
   ```

   All 11 relative imports live in the header block; there are no lazy relative
   imports inside functions.

### `krea2/transformer_krea2.py`

1. **Import rewrite** only, same mechanical rule. Verified byte-identical between the pinned
   `0.39.0` release and the `0.40.0.dev0` checkout before copying.

Not vendored on purpose:

- `autoencoder_kl_qwenimage.py` — frozen, never trained. `dflow/encoders/vae_krea2.py` wraps the
  installed class and defines encode as the exact inverse of the pipeline's decode, because
  `Krea2Pipeline` is text-to-image only and ships **no encode path** to mirror.
- `pipeline_krea2.py` — training needs no pipeline. `get_text_hidden_states` is called through a
  shim (`dflow/encoders/text_krea2.py`) rather than copied, since it is an *instance* method.

*(No behavioural changes to either vendored file.)*

## Ahead of the pin: MiniMax-H3

`minimax_h3/transformer_minimax_h3.py` is copied from **`diffusers` main @ `a949d3dd9`
(2026-08-25)**, not from the pinned release: MiniMax-H3 does not exist in 0.39.0 at all. It is a
byte-for-byte copy modulo the usual import rewrite — no behavioural changes.

Vendoring ahead of the pin is only safe because every upstream symbol the file imports was checked
to exist in the installed 0.39.0 before the copy was made:

| module | symbols |
|---|---|
| `diffusers.configuration_utils` | `ConfigMixin`, `register_to_config` |
| `diffusers.loaders` | `PeftAdapterMixin` |
| `diffusers.utils` | `BaseOutput`, `apply_lora_scale`, `logging` |
| `diffusers.models._modeling_parallel` | `ContextParallelInput`, `ContextParallelOutput` |
| `diffusers.models.attention` | `AttentionMixin`, `AttentionModuleMixin`, `FeedForward` |
| `diffusers.models.attention_dispatch` | `dispatch_attention_fn` |
| `diffusers.models.cache_utils` | `CacheMixin` |
| `diffusers.models.embeddings` | `TimestepEmbedding`, `Timesteps` |
| `diffusers.models.modeling_utils` | `ModelMixin`, `get_parameter_dtype` |

`tests/models/test_family_minimax_h3.py` instantiates the model and runs a forward against the
pinned release, so the day one of those symbols moves the suite fails rather than the model going
quietly wrong. `vendor_diff.py` lists the file under `AHEAD_OF_PIN` — there is no installed
baseline to diff it against, so it is reported rather than compared. **When the pin moves to a
release that ships MiniMax-H3, move the entry into `VENDORED` and diff it for real.**

Not vendored, and needed before H3 can train or sample:

- the audio and video VAEs, and H3's text encoder (`diffsynth-studio` has readable ports:
  `diffsynth/models/minimax_h3_{video_vae,audio_vae,text_encoder}.py`)
- the packed-layout builders in `modular_pipelines/minimax_h3/` (~4500 lines, of which
  `before_denoise.py` is the layout and `references.py` the reference dataclasses). These are
  *pipeline* code, not model code, so per rule 6 below they should be reimplemented in a task
  module rather than copied.

## Rules

1. **Never edit the `diffusers` checkout.** It is a read-only reference. All edits
   happen in this directory.
2. **Keep class names identical to upstream.** `save_pretrained` writes the class
   name into `config.json` as `_class_name`; renaming `Flux2Transformer2DModel`
   would make our checkpoints unloadable by the stock `Flux2KleinPipeline`.
3. **Prefer additive changes.** New optional `__init__` args (registered via
   `@register_to_config`) load fine against official checkpoints — the missing key
   just falls back to the default.
4. **Log every behavioural change above**, and keep the diff minimal so rebasing
   onto a newer `diffusers` stays cheap.
5. Run `python tools/checks/vendor_diff.py` to see exactly what we changed. It normalises
   the import rewrite so it does not show up as noise.
6. **Vendor model definitions only.** Pipelines, schedulers and layout builders are reimplemented
   in `tasks/`, not copied — they are where our conditioning differs from upstream's on purpose.
7. **Vendoring ahead of the pin requires a symbol audit and a forward test**, both recorded above.
   Without them the failure mode is the one this file exists to prevent: not an `ImportError`, but
   a model that runs and conditions differently.
