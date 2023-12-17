import unittest

from src.evaluator import Evaluator


class TestEvaluator(unittest.TestCase):
    def test_run_base(self):
        evaluator = Evaluator(
            model="openai/clip-vit-large-patch14",
            dataset_name="cifar10",
            dataset_split="test[0:10]",
            input_column="img",
            label_column="label",
        )
        results = evaluator.run()

        print(results)
