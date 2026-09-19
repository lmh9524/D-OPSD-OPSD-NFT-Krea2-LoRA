"""Rank-aware logging.

Only rank 0 writes. Every other rank's calls are cheap no-ops, so callers never need to guard
their own logging — which is what stops eight ranks interleaving the same line.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from dflow.config import LoggingConfig


class TrainLogger:
    def __init__(self, config: LoggingConfig, *, rank: int) -> None:
        self.config = config
        self.rank = rank
        self.enabled = rank == 0
        self.directory = Path(config.directory)
        self.writer = None
        self._logger: logging.Logger | None = None
        self._handlers: list[logging.Handler] = []

        if not self.enabled:
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        if config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.directory))
            except ImportError:
                self.warning("tensorboard is not installed; scalar logging disabled")

    def _setup_logging(self) -> None:
        logger = logging.getLogger(f"dflow.r{self.rank}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

        formatter = logging.Formatter(
            fmt=f"[%(asctime)s] %(levelname)-7s [r{self.rank}] %(message)s",
            datefmt="%H:%M:%S",
        )
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

        file_handler = logging.FileHandler(self.directory / self.config.filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self._logger = logger
        self._handlers = [console, file_handler]

    def info(self, message: str) -> None:
        if self.enabled and self._logger:
            self._logger.info(message)

    def warning(self, message: str) -> None:
        if self.enabled and self._logger:
            self._logger.warning(message)

    def error(self, message: str) -> None:
        if self.enabled and self._logger:
            self._logger.error(message)

    def log_config(self, config: Any) -> None:
        """Dump the resolved config to the run directory as well as the log.

        The file is the point: a log line scrolls away, but ``config.json`` next to the
        checkpoints answers "what was this run actually configured with" months later.
        """
        if not self.enabled:
            return
        payload = asdict(config) if is_dataclass(config) else config
        text = json.dumps(payload, indent=2, default=str, sort_keys=True)
        (self.directory / "config.json").write_text(text, encoding="utf-8")
        self.info(f"config:\n{text}")

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        if not self.enabled or step % self.config.interval:
            return
        parts = []
        for name, value in metrics.items():
            if name == "step_time":
                parts.append(f"{value:.3f}s/step")
            elif name == "peak_memory":
                parts.append(f"{value:.1f}GiB")
            else:
                parts.append(f"{name}={value:.6g}")
        self.info(f"step {step}: " + "  ".join(parts))
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
        if self._logger is not None:
            for handler in self._handlers:
                handler.flush()
                self._logger.removeHandler(handler)
                handler.close()
        self._handlers.clear()


__all__ = ["TrainLogger"]
