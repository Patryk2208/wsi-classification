from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, List, Dict

import numpy as np
from numpy import ndarray
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, cross_val_score, learning_curve, \
    RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


@dataclass
class TuningResult:
    """Your existing TuningResult - kept as is"""
    model: Any
    config: Any
    tuner: Any
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None
    cv_scores: Optional[List[float]] = None
    best_params: Optional[Dict] = None

    @property
    def model_name(self) -> str:
        if isinstance(self.model, Pipeline):
            return self.model.steps[-1][1].__class__.__name__
        return self.model.__class__.__name__

    @property
    def tuner_name(self) -> str:
        return self.tuner.__class__.__name__


class Tuner(ABC):
    """
    Performs tuning on the estimator. Returns Confusion Matrix.
    """
    def __init__(self):
        self.scaler_mapping = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

    @abstractmethod
    def tune(self, config, x, y, estimator) -> TuningResult:
        pass

def refit_strategy(cv_results):
    mean_test = cv_results["mean_test_score"]
    mean_train = cv_results["mean_train_score"]

    gap = mean_train - mean_test

    alpha = 0.5  # how strongly you penalize overfitting

    custom_score = mean_test - alpha * gap

    return np.argmax(custom_score)


class NoTuning(Tuner):
    def tune(self, config, x, y, estimator) -> TuningResult:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config.test_size, random_state=config.random_state)

        pipeline = estimator
        if config.scaling_method != "none":
            pipeline = Pipeline([
                ('scaler', self.scaler_mapping[config.scaling_method]),
                ('estimator', estimator)
            ])

        pipeline.fit(x_train, y_train)
        return TuningResult(
            model=estimator,
            config=config,
            tuner=self,
            y_true=y_test,
            y_pred=pipeline.predict(x_test),
            y_proba=pipeline.predict_proba(x_test) if hasattr(estimator, 'predict_proba') else None
        )

class CrossValidationTuning(Tuner):
    def tune(self, config, x, y, estimator) -> TuningResult:
        pipeline = estimator
        if config.scaling_method != "none":
            pipeline = Pipeline([
                ('scaler', self.scaler_mapping[config.scaling_method]),
                ('estimator', estimator)
            ])
        # if estimator.__class__.__name__ == 'StackingClassifier':
        #     return TuningResult(
        #         model=estimator,
        #         config=config,
        #         tuner=self,
        #         y_true=y,
        #         y_pred=ndarray(len(y)),
        #         y_proba=None,
        #         cv_scores=None
        #     )
        rkf = RepeatedKFold(n_splits=config.cv_folds, n_repeats=config.grid_search_repeats, random_state=config.random_state)

        y_pred = cross_val_predict(pipeline, x, y, cv=config.cv_folds, n_jobs=-1)
        y_proba = cross_val_predict(pipeline, x, y, cv=config.cv_folds, method='predict_proba', n_jobs=-1) \
            if hasattr(estimator, 'predict_proba') else None
        cv_scores = cross_val_score(pipeline, x, y, cv=rkf, scoring=config.cv_scoring_method, n_jobs=-1)

        pipeline.fit(x, y)

        return TuningResult(
            model=estimator,
            config=config,
            tuner=self,
            y_true=y,
            y_pred=y_pred,
            y_proba=y_proba,
            cv_scores=cv_scores
        )

class GridSearchTuning(Tuner):
    def tune(self, config, x, y, estimator) -> TuningResult:
        pipeline = estimator
        if config.scaling_method != "none":
            pipeline = Pipeline([
                ('scaler', self.scaler_mapping[config.scaling_method]),
                ('estimator', estimator)
            ])
        rkf = RepeatedKFold(n_splits=config.cv_folds, n_repeats=config.grid_search_repeats, random_state=config.random_state)

        n = estimator.__class__.__name__
        g = getattr(config.grid_search_params, n).param_grid
        prefixed_g = {
            f'estimator__{k}': v
            for k, v in g.items()
        }
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=prefixed_g,
            cv=rkf,
            n_jobs=-1,
            scoring=config.grid_search_scoring_method,
            return_train_score=True,
            refit=refit_strategy
        )
        grid.fit(x, y)

        y_pred = grid.predict(x)
        y_proba = grid.predict_proba(x) if hasattr(grid, 'predict_proba') else None

        return TuningResult(
            model=grid.best_estimator_,
            config=config,
            tuner=self,
            y_true=y,
            y_pred=y_pred,
            y_proba=y_proba,
            cv_scores=grid.cv_results_['mean_test_score'],
            best_params=grid.best_params_
        )

