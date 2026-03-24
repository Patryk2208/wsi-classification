from types import SimpleNamespace
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'Stacking': lambda : StackingClassifier(
                estimators=[(v, self.standard_model_mappings[v]) for v in self.config.stacking_estimators]
            )
        }

        self.standard_models = [self.standard_model_mappings[m] for m in self.config.standard_models]
        self.ensemble_models = [self.ensemble_model_mappings[m] for m in self.config.ensemble_models]

        self.evaluation_metrics_cofunctions = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'fallout': lambda y_true, y_pred: (
                cm := confusion_matrix(y_true, y_pred),
                cm[0][0] / (cm[0][0] + cm[0][1])
            ),
        }


