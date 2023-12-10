from pydantic import BaseModel
from evaluate import evaluator
from datasets import load_dataset


class Evaluator(BaseModel):
    dtaset_name: str = "cifar10"

    def __init__(self, **data):
        super().__init__(**data)

    def run(self):
        task_evaluator = evaluator("image-classification")
        data = load_dataset("beans", split="test[:40]")
        results = task_evaluator.compute(
            model_or_pipeline="nateraw/vit-base-beans",
            data=data,
            label_column="labels",
            metric="accuracy",
            label_mapping={"angular_leaf_spot": 0, "bean_rust": 1, "healthy": 2},
            strategy="bootstrap",
        )
