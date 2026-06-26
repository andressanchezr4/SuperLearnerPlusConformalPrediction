"""Progress bars and logging helpers for regression training."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that keeps tqdm progress bars readable."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            tqdm.write(message)
        except Exception:
            self.handleError(record)


def setup_logger(output_dir: str | Path, name: str = "superlearner_regression") -> logging.Logger:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(output_path / "training_log.txt", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger


def progress_bar(
    iterable: Iterable[T],
    *,
    desc: str,
    total: Optional[int] = None,
    leave: bool = True,
) -> Iterator[T]:
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, leave=leave)


@contextmanager
def timed_step(logger: logging.Logger, message: str):
    start = time.perf_counter()
    logger.info("%s...", message)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2f seconds", message, elapsed)
