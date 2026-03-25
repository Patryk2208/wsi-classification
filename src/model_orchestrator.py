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


        self.model_mappings = {
            'LogisticRegression': LogisticRegression(),
            'DecisionTree': DecisionTreeClassifier(),
            'SVM': SVC(),
            'KNN': KNeighborsClassifier(),
            'NaiveBayes': GaussianNB(),
            'RandomForest': RandomForestClassifier(),
            'GradientBoosting': GradientBoostingClassifier()
        }

        self.stacking_model = {
            'Stacking': StackingClassifier(
                estimators=[(v, self.model_mappings[v]) for v in self.config.stacking_estimators],
                final_estimator=self.model_mappings[self.config.stacking_final_estimator]
            )
        }

        self.models = [self.model_mappings[m] for m in self.config.models]
        if self.config.use_stacking:
            self.models.append(self.stacking_model['Stacking'])

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

