import fire
from src.main import Main

if __name__ == "__main__":
    fire.Fire(
        {
            "main": Main,
        }
    )
