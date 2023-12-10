from pathlib import Path
from typing import List

import numpy as np
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from pydantic import BaseModel

from src.utils.null_inversion import (
    NullInversion,
    load_img,
    make_controller,
    run_and_display,
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
