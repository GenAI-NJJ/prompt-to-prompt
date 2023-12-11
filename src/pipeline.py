from typing import List
from pydantic import BaseModel
from transformers import (
    CLIPImageProcessor,
    ImageClassificationPipeline,
)
from PIL import Image


class DiffusionImageClassifierPipeline(ImageClassificationPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(
        self,
        top_k=None,
        timeout=None,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        clip_sample: bool = False,
        set_alpha_to_one: bool = False,
        cross_replace_steps: dict = {"default_": 0.8},
        self_replace_steps: float = 0.5,
    ):
        pre, forward, post = super()._sanitize_parameters(top_k, timeout)
        forward.update(
            {
                "beta_start": beta_start,
                "beta_end": beta_end,
                "beta_schedule": beta_schedule,
                "clip_sample": clip_sample,
                "set_alpha_to_one": set_alpha_to_one,
                "cross_replace_steps": cross_replace_steps,
                "self_replace_steps": self_replace_steps,
            }
        )

        return pre, forward, post

    def _forward(
        self,
        model_inputs,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        clip_sample: bool = False,
        set_alpha_to_one: bool = False,
        cross_replace_steps: dict = {"default_": 0.8},
        self_replace_steps: float = 0.5,
    ):
        pass


class Classifier(BaseModel):
    model: str = "CompVis/stable-diffusion-v1-4"
    device: int = 0

    # DDIMScheduler params
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    clip_sample: bool = False
    set_alpha_to_one: bool = False

    # NullOptimization params
    cross_replace_steps: dict = {"default_": 0.8}
    self_replace_steps: float = 0.5

    def run(
        self,
        path: str,
        template: str,
        classes: List[str],
    ):
        pipeline = DiffusionImageClassifierPipeline(
            model=self.model,
            device=self.device,
            template=template,
            classes=classes,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            clip_sample=self.clip_sample,
            set_alpha_to_one=self.set_alpha_to_one,
            cross_replace_steps=self.cross_replace_steps,
            self_replace_steps=self.self_replace_steps,
        )

        return pipeline(Image.open(path))
