"""
This is a boilerplate pipeline 'freesound'
generated using Kedro 0.19.10
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    training_pipeline,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=training_pipeline,
            inputs=[
                "FSD50K_train_audio",
                "FSD50K_eval_audio",
                "GT_train_labels",
                "GT_eval_labels",
                "GT_vocab",
                "params:pre_processing_parameters",
                "params:training_parameters",
            ],
            outputs="FSD50K_trained_model",
            name="train_model",
        )
    ])
