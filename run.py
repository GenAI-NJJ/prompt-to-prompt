from evaluate import Evaluator
import fire
from src.classifier import Classifier

if __name__ == "__main__":
    fire.Fire(
        {
            "classifier": Classifier,
            "evaluator": Evaluator,
        }
    )
