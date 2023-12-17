from typing import List

from pydantic import BaseModel
from transformers import pipeline

""" for registering the model """
import src.model


class Classifier(BaseModel):
    model: str
    device: int

    # DDIMScheduler params
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    clip_sample: bool = False
    set_alpha_to_one: bool = False

    # NullOptimization params
    cross_replace_steps: dict = {"default_": 0.8}
    self_replace_steps: float = 0.5

    template: str = "{label}"

    def run(
        self,
        image_path: str,
        candidate_labels: List[str],
    ):
        classifier = pipeline(
            task="zero-shot-image-classification",
            model=self.model,
            device=self.device,
        )

        return classifier(
            image_path,
            candidate_labels=candidate_labels,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            clip_sample=self.clip_sample,
            set_alpha_to_one=self.set_alpha_to_one,
            cross_replace_steps=self.cross_replace_steps,
            self_replace_steps=self.self_replace_steps,
            template=self.template,
        )
