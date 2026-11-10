"""MLflow setup. Call once per process before the first request."""

from __future__ import annotations

import mlflow

from .config import Settings


def setup_tracing(settings: Settings) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    if settings.tracing_enabled:
        mlflow.tracing.enable()
    else:
        mlflow.tracing.disable()
