from types import SimpleNamespace
from typing import List

from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.tuner import GridSearchTuning, CrossValidationTuning, NoTuning, TuningResult

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

"""
Orchestrates the training and evaluation of the model.
We do not need to test different preprocessing techniques here, as the data is complete, so we will only focus on
models, evaluation metrics, and hyperparameters.
"""
class ModelOrchestrator:
    def __init__(self, config: SimpleNamespace, data: tuple[DataFrame, DataFrame]):
        self.config = config
        self.X, self.Y = data

        self.standard_model_mappings = {
            'LogisticRegression': LogisticRegression(),
            'DecisionTree': DecisionTreeClassifier(),
            'SVM': SVC(),
            'KNN': KNeighborsClassifier(),
            'NaiveBayes': GaussianNB(),
        }
        self.ensemble_model_mappings = {
            'RandomForest': RandomForestClassifier(),
            'GradientBoosting': GradientBoostingClassifier(),
            'Stacking': lambda : StackingClassifier(
                estimators=[(v, self.standard_model_mappings[v]) for v in self.config.stacking_estimators]
            )
        }

        self.standard_models = [self.standard_model_mappings[m] for m in self.config.standard_models]
        self.ensemble_models = [self.ensemble_model_mappings[m] for m in self.config.ensemble_models]
        self.models = self.standard_models + self.ensemble_models

        self.tuning_methods_mappings = {
            'None': NoTuning(),
            'CrossValidation': CrossValidationTuning(),
            'GridSearch': GridSearchTuning()
        }

        self.tuning_methods = [self.tuning_methods_mappings[t] for t in self.config.tuning_methods]


    def experiment(self) -> List[TuningResult]:
        results = []
        for model in self.models:
            print(f"Training {model.__class__.__name__}...")
            for tuner in self.tuning_methods:
                print(f"Tuning method: {tuner.__class__.__name__}")
                res = tuner.tune(self.config, self.X, self.Y, model)
                results.append(res)
        return results

