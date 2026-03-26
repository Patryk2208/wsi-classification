import pandas as pd
import yaml

from src.config_loader import load_config
from src.experiment_runner import Experiment
from src.model_orchestrator import ModelOrchestrator
from src.preprocessor import DataPreprocessor


def main():
    config_path = "../config/config.yaml"
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    config = load_config(config_path)
    df = pd.read_csv("../data/raw/ortodoncja.csv")
    pre = DataPreprocessor(df, config)

    mo = ModelOrchestrator(config, pre.preprocess())
    results = mo.experiment()
    ex = Experiment(experiment_name="15-Trying-Smaller-Stacking", common_config=raw_config)
    for result in results:
        ex.add_result(result)
    ex.generate_report(output_dir="../docs/reports")


if __name__ == "__main__":
    main()