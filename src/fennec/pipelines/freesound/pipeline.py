"""
This is a boilerplate pipeline 'freesound'
generated using Kedro 0.19.10
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    training_pipeline,
    extract_features
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=extract_features,
            inputs=[
                "FSD50K_train_audio",
                "FSD50K_eval_audio",
                "params:pre_processing_parameters",
            ],
            outputs=[
                "FSD50K_train_features",
                "FSD50K_eval_features",
            ],
            tags="feature_extraction",
        ),
        node(
           func=training_pipeline,
           inputs=[
                "FSD50K_train_features",
                "FSD50K_eval_features",
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
