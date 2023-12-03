from pathlib import Path
from typing import List
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionPipeline
import numpy as np

from pydantic import BaseModel

from null_inversion import NullInversion, load_img, make_controller, run_and_display


class Main(BaseModel):
    src_prompt: str
    src_img_path: str
    tgt_prompts: List[str]
    device: str = "cuda:3"
    model_name: str = "CompVis/stable-diffusion-v1-4"
    output_dir: str = "output"
    img_rescale_size: int = 512

    def __init__(self, **data):
        super().__init__(**data)

        # prepare output dir
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run(self):
        src_img = load_img(self.src_img_path, img_rescale_size=self.img_rescale_size)

        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

        ldm_stable = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            scheduler=scheduler,
        ).to(self.device)

        null_inversion = NullInversion(ldm_stable)

        _, x_t, uncond_embeddings = null_inversion.invert(
            src_img,
            self.src_prompt,
            verbose=True,
        )

        cross_replace_steps = {"default_": 0.8}
        self_replace_steps = 0.5

        tgt_imgs = []
        tgt_latents = []
        for tgt_prompt in self.tgt_prompts:
            controller = make_controller(
                prompts=[self.src_prompt, tgt_prompt],
                is_replace_controller=False,
                cross_replace_steps=cross_replace_steps,
                self_replace_steps=self_replace_steps,
            )
            (images, latents), x_t = run_and_display(
                [self.src_prompt, tgt_prompt],
                controller,
                run_baseline=False,
                latent=x_t,
                uncond_embeddings=uncond_embeddings,
            )
            back_translate_img = images[0]
            tgt_imgs.append(images[-1])
            tgt_latents.append(latents[-1])

        # save images
        images = [src_img, back_translate_img] + tgt_imgs
        save_images(images, self.output_dir, "images", num_rows=1, offset_ratio=0.02)

        # print errors
        print("Errors using L2 norm:")
        for i, latent in enumerate(tgt_latents):
            print(
                f"{self.tgt_prompts[i]}: {np.linalg.norm(x_t - latent.cpu().numpy())}"
            )


def save_images(images, output_dir, filename, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = (
        np.ones(
            (
                h * num_rows + offset * (num_rows - 1),
                w * num_cols + offset * (num_cols - 1),
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )
    for i in range(num_rows):
        for j in range(num_cols):
            image_[
                i * (h + offset) : i * (h + offset) + h :,
                j * (w + offset) : j * (w + offset) + w,
            ] = images[i * num_cols + j]

    pil_img = Image.fromarray(image_)
    pil_img.save(output_dir / f"{filename}.png")


if __name__ == "__main__":
    main = Main(
        src_prompt="",
        src_img_path="example_images/gnochi_mirror.jpeg",
        tgt_prompts=["dog", "cat", "bird"],
        img_rescale_size=512,
    )
    main.run()
