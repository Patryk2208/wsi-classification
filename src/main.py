import pandas as pd

from src.config_loader import load_config
from src.model_orchestrator import ModelOrchestrator
from src.preprocessor import DataPreprocessor


def main():
    config = load_config("../config/config.yaml")
    df = pd.read_csv("../data/raw/ortodoncja.csv")
    pre = DataPreprocessor(df, config)

    mo = ModelOrchestrator(config, pre.preprocess())
    mo.experiment()


if __name__ == "__main__":
    main()