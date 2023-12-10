from pathlib import Path
from typing import List

import torch
import numpy as np
from diffusers import DDIMScheduler, StableDiffusionPipeline
from pydantic import BaseModel

from src.utils.image_saver import save_images
from src.utils.null_inversion import (
    NullInversion,
    load_img,
    make_controller,
    run_and_display,
)


class Inverter(BaseModel):
    # src params
    src_img_path: str
    src_prompt: str
    img_rescale_size: int = 512

    # model params
    devices: List[int] = [0]
    model_name: str = "CompVis/stable-diffusion-v1-4"

    # DDIMScheduler params
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    clip_sample: bool = False
    set_alpha_to_one: bool = False

    # NullOptimization params
    cross_replace_steps: dict = {"default_": 0.8}
    self_replace_steps: float = 0.5

    def __init__(self, **data):
        super().__init__(**data)

        # prepare output dir
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # load src image
        self.src_img = load_img(
            self.src_img_path,
            img_rescale_size=self.img_rescale_size,
        )

    def _null_invert(self):
        scheduler = DDIMScheduler(
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            clip_sample=self.clip_sample,
            set_alpha_to_one=self.set_alpha_to_one,
        )

        ldm_stable = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            scheduler=scheduler,
        ).to(self.device)

        null_inversion = NullInversion(ldm_stable)

        image_rec, x_t, uncond_embeddings = null_inversion.invert(
            self.src_img,
            self.src_prompt,
            verbose=True,
        )

        return image_rec, x_t, uncond_embeddings

    def _prompt_to_prompt(self, tgt_prompt, uncond_embeddings):
        controller = make_controller(
            prompts=[self.src_prompt, tgt_prompt],
            is_replace_controller=False,
            cross_replace_steps=self.cross_replace_steps,
            self_replace_steps=self.self_replace_steps,
        )
        (images, latents), x_t = run_and_display(
            [self.src_prompt, tgt_prompt],
            controller,
            run_baseline=False,
            latent=x_t,
            uncond_embeddings=uncond_embeddings,
        )

        return images[0], latents[-1], latents[-1]

    def run_one(self, tgt_prompt: str):
        image_rec, x_t, uncond_embeddings = self._null_invert()
        back_translate_img, tgt_img, tgt_latent = self._prompt_to_prompt(
            tgt_prompt,
            uncond_embeddings,
        )

        return back_translate_img, tgt_img, tgt_latent

    def run(self, tgt_prompts: List[str]):
        image_rec, x_t, uncond_embeddings = self._null_invert()

        tgt_imgs = []
        tgt_latents = []
        for tgt_prompt in tgt_prompts:
            back_translate_img, tgt_img, tgt_latent = self._prompt_to_prompt(
                tgt_prompt,
                uncond_embeddings,
            )

            tgt_imgs.append(tgt_img)
            tgt_latents.append(tgt_latent)

        # save images
        images = [self.src_img, back_translate_img] + tgt_imgs

        save_images(images, self.output_dir, "images", num_rows=1, offset_ratio=0.02)

        # print errors
        print("Errors using L2 norm:")
        for i, latent in enumerate(tgt_latents):
            print(
                f"{self.tgt_prompts[i]}: {np.linalg.norm(x_t - latent.cpu().numpy())}"
            )
