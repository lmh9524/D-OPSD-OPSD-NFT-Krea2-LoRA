# Krea 2 multi-image reference: what is actually being trained

Short version: **Krea 2 has no pretrained reference conditioning.** Adding it to this framework was
integration work; making it *work* is a training problem that has not been solved yet, and this
document exists so nobody mistakes the first for the second.

## The finding

`Krea2Pipeline` is text-to-image only. Its whole image-side id construction is five lines:

```python
# diffusers/pipelines/krea2/pipeline_krea2.py
@staticmethod
def prepare_position_ids(text_seq_len, grid_height, grid_width, device):
    text_ids = torch.zeros(text_seq_len, 3, device=device)
    image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
    image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
    image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
    return torch.cat([text_ids, image_ids.reshape(-1, 3)], dim=0)
```

Axis 0 — the `T` axis — is never written. It is allocated 32 of the 128 head dimensions by
`axes_dims_rope = (32, 48, 48)`, and it has held the value zero for every token the model has ever
seen, in pretraining and in every published checkpoint.

There is also no second image span. The sequence is `[text; image]`, one image, always.

## Why that is different from FLUX.2

FLUX.2 klein looks superficially similar — same trick, references offset along `T` — but klein was
**trained that way**. `Flux2KleinPipeline` concatenates reference latents in its denoising loop and
`_prepare_image_ids` puts reference *i* at `T = 10 * (i + 1)`. A LoRA on klein adjusts how an
existing capability behaves.

|  | FLUX.2 klein | Krea 2 |
|---|---|---|
| reference spans in the sequence | pretrained | never seen |
| non-zero `T` coordinate | pretrained | never seen |
| what a LoRA does | adapts a capability | teaches a new one |
| realistic rank | 16–32 | 64+, possibly full fine-tune |
| realistic data | hundreds of pairs | tens of thousands |

The layout we build for Krea 2 (`build_krea2_sequence`) is the same one FLUX.2 and Qwen-Image-Edit
use, and it is structurally sound — the rotary axis exists, the model has no shape constraint that
forbids a longer sequence, and no weight surgery is needed. What is missing is any reason to expect
the weights to *mean* anything at `T = 10` on step zero.

## What to expect

* Early samples will ignore the references completely. That is not a bug in the conditioning; it is
  the model having no prior that those tokens are informative.
* The loss will fall anyway, because the text-conditioned part of the task still works. **Loss is not
  a signal of whether references are being used.** Judge it by sampling with and without references
  and comparing, not by the curve.
* If it plateaus while ignoring references, the next things to try, in order: raise rank, unfreeze
  `img_in` and the first blocks, then full fine-tune. `attn.to_gate` is in the default LoRA targets
  precisely because it is the gate that decides how much attention output each channel admits.

## Why `Krea-2-Raw` rather than `Krea-2-Turbo`

The recipe defaults to `krea2-raw`. Turbo is the few-step TDM-distilled checkpoint: its trajectory
has been deliberately collapsed toward 4–8 step sampling, and fine-tuning on top of that adapts the
collapsed trajectory rather than the underlying field. This is the same reason `registry.py` calls
klein-**base** "the variant BFL recommends for fine-tuning".

If Turbo is what must ship, train it anyway — but note that its pipeline pins `mu = 1.15` at every
resolution, and `FlowMatchConfig.shift` defaults to `exp(1.15) = 3.1582`. That is not a coincidence
worth relying on silently: it means the default fixed schedule matches Turbo's inference schedule
exactly, and `--flow-match.shift None` would *break* that match rather than improve it.

## The tripwire

`tests/test_upstream_contract.py::test_krea2_pipeline_pins_the_t_axis_at_zero` asserts that upstream
still leaves `image_ids[..., 0]` unwritten. If Krea ever ships native reference conditioning, that
test fails, and the right response is to delete our hand-built ids and use theirs.
