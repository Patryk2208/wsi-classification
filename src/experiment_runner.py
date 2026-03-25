import mlflow
import mlflow.sklearn
import yaml
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from src.tuner import TuningResult


class Experiment:
    """
    Manages MLflow experiment with a list of TuningResults.
    Each TuningResult becomes a separate MLflow run.
    """

    def __init__(self,
                 experiment_name: str,
                 common_config: dict,
                 tracking_uri: str = '../experiments',
                 save_artifacts: bool = True):
        """
        Args:
            experiment_name: Name for this experiment (used in MLflow)
            common_config: Shared configuration across all tuning results
            tracking_uri: MLflow tracking URI (default: ./experiments)
            save_artifacts: Whether to save plots and reports
        """
        self.experiment_name = experiment_name
        self.common_config = common_config
        self.save_artifacts = save_artifacts
        self.results: List[TuningResult] = []

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.run_metadata = []

        self.scoring_methods_mappings = {
            'accuracy': self.my_accuracy_score,
            'precision': self.my_precision_score,
            'recall': self.my_recall_score,
            'confusion_matrix': self.my_confusion_matrix,
            'f1': self.my_f1_score,
            'roc_auc': self.my_roc_auc,
            'custom': self.my_custom_scoring_method
        }

        print(f"MLflow Experiment initialized: {experiment_name}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    def add_result(self, result: TuningResult) -> str:
        """
        Add a TuningResult and log it as an MLflow run.
        Returns the run_id.
        """
        self.results.append(result)

        with mlflow.start_run(run_name=f"{result.model_name}_{result.tuner_name}") as run:
            run_id = run.info.run_id

            self._log_config()

            mlflow.log_param("model", result.model_name)
            mlflow.log_param("tuner", result.tuner_name)

            if result.best_params:
                for param_name, param_value in result.best_params.items():
                    mlflow.log_param(f"best_{param_name}", param_value)

            metrics = self._compute_metrics(result)
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)

            if result.cv_scores is not None:
                mlflow.log_metric("cv_mean", np.mean(result.cv_scores))
                mlflow.log_metric("cv_std", np.std(result.cv_scores))
                mlflow.log_text(str(result.cv_scores), "cv_scores.txt")

            if self.save_artifacts:
                self._log_confusion_matrix(result)
                self._log_roc_curve(result)
                self._log_classification_report(result)

            mlflow.sklearn.log_model(result.model, name="model")

            self.run_metadata.append({
                'run_id': run_id,
                'model': result.model_name,
                'tuner': result.tuner_name,
                'metrics': metrics,
                'best_params': result.best_params,
                'cv_scores': result.cv_scores
            })

            print(f"Logged: {result.model_name} | {result.tuner_name} | Accuracy: {metrics.get('accuracy', 'N/A'):.4f} | F1: {metrics.get('f1', 'N/A'):.4f}")

            return run_id

    def _log_config(self):
        """Log configuration as artifact and parameters"""

        for key, value in self.common_config.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(f"config_{key}", value)

        mlflow.log_dict(self.common_config, "config.yaml")

    def _evaluate(self, r: TuningResult):


        print()

    def _compute_metrics(self, result: TuningResult) -> Dict[str, float]:
        """Compute all evaluation metrics"""
        metrics = {}

        for sm in result.config.scoring_methods:
            s = self.scoring_methods_mappings[sm](self, result)
            metrics[sm] = s

        return metrics

    def _log_confusion_matrix(self, result: TuningResult):
        """Generate and log confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(result.y_true, result.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix: {result.model_name} ({result.tuner_name})')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close()

    def _log_roc_curve(self, result: TuningResult):
        """Generate and log ROC curve"""
        if result.y_proba is None:
            return

        y_true, y_proba = result.y_true, result.y_proba
        n_classes = len(np.unique(y_true))

        fig, ax = plt.subplots(figsize=(8, 6))

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_title(f'ROC Curve: {result.model_name}')
        else:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=np.unique(y_true))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_title(f'ROC Curves (OvR): {result.model_name}')

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')

        mlflow.log_figure(fig, "roc_curve.png")
        plt.close()

    def _log_classification_report(self, result: TuningResult):
        """Log classification report as text"""
        report = classification_report(result.y_true, result.y_pred)
        mlflow.log_text(report, "classification_report.txt")

    def generate_report(self,
                        hitl_notes: str = "",
                        output_dir: str = "../docs/reports") -> Path:
        """
        Generate comprehensive markdown report with HITL section.
        Call this after all results are added.
        """
        output_path = Path(output_dir) / f"{self.experiment_name}_report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            # Header
            f.write(f"# Experiment Report: {self.experiment_name}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # MLflow UI Link
            f.write("## View in MLflow\n\n")
            f.write("Run the following command and navigate to this experiment:\n")
            f.write("```bash\nmlflow ui\n```\n\n")
            f.write(f"**Experiment Name:** `{self.experiment_name}`\n\n")
            f.write("---\n\n")

            # HITL Section
            f.write("## HITL (Human-in-the-Loop) Notes\n\n")
            if hitl_notes:
                f.write(f"{hitl_notes}\n\n")
            else:
                f.write("*No notes provided. Add observations, hypotheses, and next steps.*\n\n")
            f.write("---\n\n")

            # Results Summary Table
            f.write("## Results Summary\n\n")
            f.write("| # | Model | Tuner | Accuracy | Precision | Recall | F1 | ROC-AUC |\n")
            f.write("|---|-------|-------|----------|-----------|--------|-----|---------|\n")

            for i, meta in enumerate(self.run_metadata, 1):
                m = meta['metrics']
                f.write(f"| {i} | {meta['model']} | {meta['tuner']} | ")
                f.write(f"{m.get('accuracy', 'N/A'):.4f} | ")
                f.write(f"{m.get('precision', 'N/A'):.4f} | ")
                f.write(f"{m.get('recall', 'N/A'):.4f} | ")
                f.write(f"{m.get('f1', 'N/A'):.4f} | ")
                f.write(f"{m.get('roc_auc', 'N/A'):.4f} |\n")

            f.write("\n---\n\n")

            # Best Model Selection
            if self.run_metadata:
                best_by_f1 = max(self.run_metadata, key=lambda x: x['metrics'].get('f1', 0))
                best_by_acc = max(self.run_metadata, key=lambda x: x['metrics'].get('accuracy', 0))

                f.write("## Best Model Selection\n\n")
                f.write(f"**Recommended Model:** `{best_by_f1['model']}` with `{best_by_f1['tuner']}`\n\n")
                f.write("### Top Performers\n\n")
                f.write(
                    f"- **Best by F1 Score:** {best_by_f1['model']} ({best_by_f1['tuner']}) - F1: {best_by_f1['metrics']['f1']:.4f}\n")
                f.write(
                    f"- **Best by Accuracy:** {best_by_acc['model']} ({best_by_acc['tuner']}) - Acc: {best_by_acc['metrics']['accuracy']:.4f}\n\n")

                if best_by_f1['best_params']:
                    f.write("**Best Hyperparameters:**\n```json\n")
                    f.write(json.dumps(best_by_f1['best_params'], indent=2))
                    f.write("\n```\n\n")

                # MLflow run links
                f.write("### View in MLflow\n\n")
                f.write(
                    f"- [Run {best_by_f1['run_id']}]({mlflow.get_tracking_uri()}/#/experiments/.../runs/{best_by_f1['run_id']})\n\n")

            # Configuration
            f.write("## ⚙️ Common Configuration\n\n")
            f.write("```yaml\n")
            config_dict = self.common_config
            f.write(yaml.dump(config_dict, default_flow_style=False))
            f.write("```\n\n")

            # Next Steps
            f.write("## Next Steps\n\n")
            f.write("- [ ] Fine-tune best model with narrower hyperparameter ranges\n")
            f.write("- [ ] Perform feature importance analysis\n")
            f.write("- [ ] Test on holdout validation set\n")
            f.write("- [ ] Error analysis on misclassified samples\n")
            f.write("- [ ] Consider ensemble methods if performance gap exists\n")
            f.write("- [ ] Deploy model if metrics meet business requirements\n\n")

            f.write("---\n\n")
            f.write(f"*Report generated by Experiment Framework on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        print(f"\n Report generated: {output_path}")
        return output_path

    def get_best_result(self, metric: str = 'f1') -> TuningResult:
        """Get the best TuningResult by specified metric"""
        if not self.results:
            return None

        best_idx = max(range(len(self.results)),
                       key=lambda i: self.run_metadata[i]['metrics'].get(metric, 0))
        return self.results[best_idx]

    def to_dataframe(self) -> pd.DataFrame:
        """Return all results as a DataFrame for easy analysis"""
        rows = []
        for meta in self.run_metadata:
            row = {
                'run_id': meta['run_id'],
                'model': meta['model'],
                'tuner': meta['tuner']
            }
            row.update(meta['metrics'])
            rows.append(row)
        return pd.DataFrame(rows)

    def compare_runs(self, metric: str = 'f1', top_k: int = 5) -> pd.DataFrame:
        """Get top K runs by specified metric"""
        df = self.to_dataframe()
        return df.nlargest(top_k, metric)


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


# Optional: Cleanup function
def cleanup_old_experiments(keep_last_n: int = 10, tracking_uri: str = None):
    """
    Delete old experiments to save space.
    Use with caution - only if you don't need historical runs.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    experiments = client.search_experiments()
    # Sort by last_update_time
    experiments.sort(key=lambda x: x.last_update_time, reverse=True)

    for exp in experiments[keep_last_n:]:
        if exp.name.startswith("orthodontics"):  # Only delete our experiments
            print(f"Deleting old experiment: {exp.name}")
            client.delete_experiment(exp.experiment_id)