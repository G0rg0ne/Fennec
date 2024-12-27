"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.mfcc import pipeline as audio_ingestion_pipeline
from .pipelines.freesound import pipeline as FSD50K_pre_processing


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    #pipelines["__default__"] = sum(pipelines.values())
    return {
        "__default__": sum(pipelines.values()),
        "audio_ingest": audio_ingestion_pipeline.create_pipeline(),
        "FSD50K_pre_processing": FSD50K_pre_processing.create_pipeline(),
    }