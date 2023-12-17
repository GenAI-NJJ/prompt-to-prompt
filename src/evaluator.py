from pydantic import BaseModel
from evaluate import ImageClassificationEvaluator, evaluator
from datasets import load_dataset
from transformers import pipeline

""" for registering the model """
import src.model


class Evaluator(BaseModel):
    model: str = "CompVis/stable-diffusion-v1-4"
    device: int = 0

    dataset_name: str = "cifar10"
    dataset_split: str = "test"
    input_column: str = "img"
    label_column: str = "label"

    def run(self):
        data = load_dataset(
            self.dataset_name,
            split=self.dataset_split,
        )
        classifier = src.model.AdapterPipeline(
            pipeline=pipeline(
                task="zero-shot-image-classification",
                model=self.model,
                device=self.device,
            ),
            candidate_labels=data[self.label_column],
        )

        task_evaluator = evaluator("image-classification")
        results = task_evaluator.compute(
            model_or_pipeline=classifier,
            data=data,
            metric="accuracy",
            input_column="img",
            label_column=self.label_column,
            label_mapping=None,
        )

        return results
