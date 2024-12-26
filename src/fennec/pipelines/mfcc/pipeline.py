"""
This is a boilerplate pipeline 'mfcc'
generated using Kedro 0.19.10
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    monitoring_signal
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=monitoring_signal,
            inputs="input_audio",
            outputs=[
                "temporal_audio_figures",
                "librosa_spectrum_figures",
            ],
            name="direct_ingestion",
        )
    ])
