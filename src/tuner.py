from abc import ABC, abstractmethod

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, cross_val_score


class TuningResult:
    def __init__(self, model, config, tuner, y_true, y_pred, y_proba = None, cv_scores = None, best_params = None):
        self.model = model
        self.config = config
        self.tuner = tuner
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.cv_scores = cv_scores
        self.best_params = best_params

class Tuner(ABC):
    """
    Performs tuning on the estimator. Returns Confusion Matrix.
    """
    @abstractmethod
    def tune(self, config, x, y, estimator) -> TuningResult:
        pass

class NoTuning(Tuner):
    def tune(self, config, x, y, estimator) -> TuningResult:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config.test_size, random_state=config.random_state)
        estimator.fit(x_train, y_train)
        return TuningResult(
            model=estimator,
            config=config,
            tuner=self,
            y_true=y_test,
            y_pred=estimator.predict(x_test),
            y_proba=estimator.predict_proba(x_test) if hasattr(estimator, 'predict_proba') else None
        )

class CrossValidationTuning(Tuner):
    def tune(self, config, x, y, estimator) -> TuningResult:
        y_pred = cross_val_predict(estimator, x, y, cv=config.cv_folds)
        y_proba = cross_val_predict(estimator, x, y, cv=config.cv_folds, method='predict_proba') \
            if hasattr(estimator, 'predict_proba') else None
        cv_scores = cross_val_score(estimator, x, y, cv=config.cv_folds, scoring=config.cv_scoring_method)

        estimator.fit(x, y)

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
        n = estimator.__class__.__name__
        g = getattr(config.grid_search_params, n).param_grid
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=g,
            cv=config.cv_folds,
            n_jobs=-1,
            scoring=config.grid_search_scoring_method
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

