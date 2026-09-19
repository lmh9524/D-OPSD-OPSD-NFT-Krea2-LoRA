# Experiment records

Each record directory contains:

- `config.json`: resolved runtime configuration; treat this as authoritative when it differs from recipe or launcher defaults.
- `train.log`: captured training log.
- `provenance.json`: Git revision, dirty-tree patch, and software/hardware environment.

The records intentionally do not contain datasets, model weights, checkpoints, generated previews, or credentials. Absolute paths describe the original machine and must be overridden for a new run.
