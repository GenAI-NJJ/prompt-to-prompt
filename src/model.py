from typing import Tuple

from transformers import (
    AutoConfig,
    AutoModelForZeroShotImageClassification,
    CLIPConfig,
    CLIPModel,
    ImageClassificationPipeline,
    PretrainedConfig,
)
from transformers.models.clip.modeling_clip import CLIPOutput


class DiffusionClassifierConfig(CLIPConfig):
    model_type = "diffusion_classifier"

    def __init__(
        self,
        model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
        template: str = "{label}",
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.template = template

        super().__init__(**kwargs)


class DiffusionClassifierModel(CLIPModel):
    config_class: PretrainedConfig = DiffusionClassifierConfig

    def __init__(self, config: DiffusionClassifierConfig):
        super().__init__(config)

        raise NotImplementedError

    def forward(
        self,
        input_ids,
        pixel_values,
    ) -> Tuple | CLIPOutput:
        raise NotImplementedError


class AdapterPipeline(ImageClassificationPipeline):
    def __init__(
        self,
        pipeline,
        candidate_labels,
    ):
        self.pipeline = pipeline
        self.task = "image-classification"
        self.candidate_labels = candidate_labels

    def __call__(self, input_texts, **kwargs):
        return self.pipeline(
            input_texts,
            candidate_labels=self.candidate_labels,
            **kwargs,
        )


AutoConfig.register(
    "diffusion_classifier",
    DiffusionClassifierModel,
)

AutoModelForZeroShotImageClassification.register(
    DiffusionClassifierConfig,
    DiffusionClassifierModel,
)
