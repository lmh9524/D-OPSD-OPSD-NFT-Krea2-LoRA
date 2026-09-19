# tests/

Laid out to mirror `dflow/`, so the test for a module is where you would look for it.

| directory | covers |
|---|---|
| `encoders/` | text and VAE encoders, the Krea 2 text shim, the embedding cache |
| `models/` | family adapters (FLUX.2, Krea 2), LoRA attach/load, real checkpoint configs |
| `tasks/` | ref2img conditioning and dataset, bucketing, retry |
| `trainer/` | the fit loop, train state, losses, the flow-matching scheduler |
| `parallel/` | mesh construction, DP/CP wiring, context-parallel equivalence and seams |
| `rewards/` | the pixel contract, the composite, the OCR normalisation. `test_aesthetic_real.py` is `slow` — it downloads CLIP |
| `rl/` | the SDE rollout and its replay. Driven by an analytic velocity field, so no diffusers and no checkpoint; its own `conftest.py` holds the fake sequence |
| (root) | `test_layering.py` and `test_upstream_contract.py` — repo-level invariants, not tied to one module, plus `conftest.py` |

```bash
pytest                 # the fast suite
pytest -m slow         # the ones that load multi-GB weights
```

`conftest.py` stays at the root so its fixtures apply to every subdirectory.
