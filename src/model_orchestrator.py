from types import SimpleNamespace
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
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

        self.scoring_methods_mappings = {
            'accuracy': self.my_accuracy_score,
            'precision': self.my_precision_score,
            'recall': self.my_recall_score,
            'confusion_matrix': self.my_confusion_matrix,
            'f1': self.my_f1_score,
            'roc_auc': self.my_roc_auc,
            'custom': self.my_custom_scoring_method
        }


    def experiment(self):
        results = []
        for model in self.models:
            print(f"Training {model.__class__.__name__}...")
            for tuner in self.tuning_methods:
                print(f"Tuning method: {tuner.__class__.__name__}")
                res = tuner.tune(self.config, self.X, self.Y, model)
                results.append(res)
        for r in results:
            self.evaluate(r)

    def evaluate(self, r: TuningResult):
        print(f"Model: {r.model.__class__.__name__} - Tuning method: {r.tuner.__class__.__name__}")
        for sm in self.config.scoring_methods:
            s = self.scoring_methods_mappings[sm](self, r)
            print(f"Scoring method: {sm}: {s}")
        print()

    @staticmethod
    def my_accuracy_score(self, r: TuningResult):
        return accuracy_score(r.y_true, r.y_pred)

    @staticmethod
    def my_f1_score(self, r: TuningResult):
        return f1_score(r.y_true, r.y_pred, average='weighted')

    @staticmethod
    def my_confusion_matrix(self, r: TuningResult):
        return confusion_matrix(r.y_true, r.y_pred)

    @staticmethod
    def my_precision_score(self, r: TuningResult):
        return precision_score(r.y_true, r.y_pred, average='weighted')

    @staticmethod
    def my_recall_score(self, r: TuningResult):
        return recall_score(r.y_true, r.y_pred, average='weighted')

    @staticmethod
    def my_roc_auc(self, r: TuningResult):
        if r.y_proba is None:
            return None
        return roc_auc_score(y_true=r.y_true, y_score=r.y_proba, multi_class='ovr')

    @staticmethod
    def my_custom_scoring_method(self, r: TuningResult):
        pass
