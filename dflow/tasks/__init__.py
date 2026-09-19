"""L4: task semantics, one folder per task.

Imported explicitly by ``experiments/`` — ``from dflow.tasks.ref2img import ...``. Task namespaces
are the one exception to "experiments import only the dflow top level", because re-exporting every
task centrally would force every task's dependencies on every run.
"""

from dflow.tasks.base import validate_batch

__all__ = ["validate_batch"]
