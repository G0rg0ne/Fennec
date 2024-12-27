"""
This is a boilerplate pipeline 'freesound'
generated using Kedro 0.19.10
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    pre_processing,
    create_datamodule
)

def create_pipeline(**kwargs) -> Pipeline:
    audio_pipeline = Pipeline(
        [
            node(
                func=create_datamodule,
                name="create_train_datamodule",
                inputs=[
                    "train_dataset",
                    "params:train_datamodule",
                ],
                outputs="train_datamodule",
                tags=[
                    "ccvpe",
                    "training",
                    "train_datamodule_creatation",
                ],
            ),
        ]
    )
    return audio_pipeline
