"""
This is a boilerplate pipeline 'freesound'
generated using Kedro 0.19.10
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    pre_process,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=pre_process,
            inputs=[
                "FSD50K_train_audio",
                "FSD50K_eval_audio",
                "GT_train_labels",
                "GT_eval_labels",
                "GT_vocab",
                "params:pre_processing_parameters",
            ],
            outputs="trained_model",
            name="train_model",
        )
    ])
