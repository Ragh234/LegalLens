import logging
import sys
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator
import json

from config.settings import settings


def configure_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def log_event(logger: logging.Logger, event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str))


@contextmanager
def timed_step(logger: logging.Logger, step_name: str) -> Iterator[None]:
    start = perf_counter()
    log_event(logger, "step_started", step=step_name)
    try:
        yield
    except Exception as exc:
        elapsed_ms = (perf_counter() - start) * 1000
        log_event(logger, "step_failed", step=step_name, latency_ms=round(elapsed_ms, 2), error=str(exc))
        raise
    finally:
        elapsed_ms = (perf_counter() - start) * 1000
        log_event(logger, "step_completed", step=step_name, latency_ms=round(elapsed_ms, 2))
