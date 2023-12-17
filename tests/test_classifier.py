import unittest

from src.classifier import Classifier


class TestClassifier(unittest.TestCase):
    def test_run_base(self):
        classifier = Classifier(
            model="openai/clip-vit-large-patch14",
            device=0,
        )
        image_path = "example_images/dog4.jpeg"
        candidate_labels = ["dog", "cat"]
        result = classifier.run(image_path, candidate_labels)

        print(result)

    def test_run_diffusion(self):
        classifier = Classifier(
            model="CompVis/stable-diffusion-v1-4",
            device=0,
        )
        image_path = "example_images/dog4.jpeg"
        candidate_labels = ["dog", "cat"]
        result = classifier.run(image_path, candidate_labels)

        print(result)


if __name__ == "__main__":
    unittest.main()
