from typing import Tuple

from diffusers import DDIMScheduler, StableDiffusionPipeline
from transformers import (
    AutoConfig,
    AutoModelForZeroShotImageClassification,
    CLIPConfig,
    CLIPModel,
    ImageClassificationPipeline,
    PretrainedConfig,
)
from transformers.models.clip.modeling_clip import CLIPOutput

from .inverter_new import NullInversion


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

        self.scheduler = DDIMScheduler(
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            set_alpha_to_one=config.set_alpha_to_one,
        )

        self.ldm_satble = StableDiffusionPipeline.from_pretrained(
            self.model_name_or_path,
            scheduler=self.scheduler,
        )

        self.unet = self.ldm_satble.unet
        self.vae = self.ldm_satble.vae
        self.tokenizer = self.ldm_satble.tokenizer
        self.text_encoder = self.ldm_satble.text_encoder
        self.scheduler = self.ldm_satble.scheduler

        self.scheduler.set_timesteps(config.num_ddim_steps)

        self.inverter = NullInversion(
            self.unet,
            self.vae,
            self.tokenizer,
            self.text_encoder,
            self.scheduler,
        )

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
