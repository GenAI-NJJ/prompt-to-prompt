from typing import List

import numpy as np
from pydantic import BaseModel
from .utils.image_saver import save_images


class Logger(BaseModel):
    output_dir: str

    def __init__(self, **data):
        super().__init__(**data)

    def save_image(
        self,
        images: List[np.ndarray],
        filename: str,
    ):
        save_images(images, self.output_dir, filename, num_rows=1, offset_ratio=0.02)
