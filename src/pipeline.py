from transformers import ImageClassificationPipeline


class DiffusionImageClassificationPipeline(ImageClassificationPipeline):
    def _forward(self, model_inputs):
        raise NotImplementedError
