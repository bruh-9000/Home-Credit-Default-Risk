from src.utils.modeling import (
    train_pipeline,
    predict_pipeline,
    prepare_data
)
from src.utils.config import (
    label,
    value_mappings,
    model_configs
)
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

np.random.seed(42)


def evaluate_pipeline(model_name, data):
    _, X_train, y_train, X_val, y_val = data

    label_mapping = value_mappings.get(label)
    if label_mapping:
        y_train = y_train.map(label_mapping)
        if y_val is not None:
            y_val = y_val.map(label_mapping)

    has_val_labels = y_val is not None

    pipe, search = train_pipeline(model_name, X_train, y_train)

    cv_metrics = {}
    if search is not None:
        idx = search.best_index_
        results = search.cv_results_

        print(f"\n{model_name} - CV METRICS (mean ± std):")
        for raw_key in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
            label_out = raw_key.capitalize()
            mean = results[f'mean_test_{raw_key}'][idx]
            std = results[f'std_test_{raw_key}'][idx]
            print(f"{label_out}: {mean:.2%} ± {std:.2%}")
            cv_metrics[label_out] = mean

        metrics = cv_metrics

    if has_val_labels:
        y_preds, y_probs = predict_pipeline(pipe, X_val)
        print(f"\n{model_name} - VAL METRICS:")
        val_metrics = get_analysis(y_val, y_preds, y_probs)

        config = model_configs.get(model_name, {})
        if not config.get('baseline', False):
            generate_graphs(pipe, X_val, y_val, y_preds, y_probs)

        if search is None:
            metrics = val_metrics

    return pipe, metrics


def get_analysis(y_var, y_preds, y_probs):
    accuracy = accuracy_score(y_var, y_preds)
    f1 = f1_score(y_var, y_preds)
    recall = recall_score(y_var, y_preds)
    precision = precision_score(y_var, y_preds)
    roc_auc = roc_auc_score(y_var, y_probs)

    report = (
        f"Accuracy: {accuracy:.2%}\n"
        f"F1 Score: {f1:.2%}\n"
        f"Precision: {precision:.2%}\n"
        f"Recall: {recall:.2%}\n"
        f"ROC AUC: {roc_auc:.2%}"
    )

    values = {
        'Accuracy': accuracy,
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
        'Roc_auc': roc_auc
    }

    print(report)

    return values


def generate_graphs(model, X_val, y_val, y_pred, y_probs):
    # Combine the three graphs into one image
    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_confusion_matrix(y_val, y_pred, ax=axes[0])
    plot_roc_curve(y_val, y_probs, ax=axes[1])
    plot_pr_curve(y_val, y_probs, ax=axes[2])

    plt.tight_layout()
    plt.show()

    plot_shap_summary(model, X_val, y_val)


def plot_confusion_matrix(y_val, y_pred, ax):
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    labels = ['Negative', 'Positive']
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)


def plot_roc_curve(y_val, y_probs, ax):
    fpr, tpr, _ = roc_curve(y_val, y_probs)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(True)


def plot_pr_curve(y_val, y_probs, ax):
    precision, recall, _ = precision_recall_curve(y_val, y_probs)
    ax.plot(recall, precision)
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid(True)


def plot_shap_summary(pipe, X_val, y_val):
    model = pipe.named_steps['model']

    X_transformed, _ = prepare_data(X_val, None, y_val)
    X_sample = X_transformed.iloc[:100]

    explainer = shap.TreeExplainer(
        model, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X_sample)[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    normalized_shap = mean_abs_shap / mean_abs_shap.max()

    sorted_idx = np.argsort(normalized_shap)[::-1][:10]
    sorted_labels = X_sample.columns[sorted_idx]
    sorted_values = normalized_shap[sorted_idx]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(10), sorted_values, color='steelblue')
    plt.yticks(range(10), sorted_labels)
    plt.xlabel('Normalized SHAP Importance (0-1)')
    plt.title('Top 10 Feature Importances (SHAP)')

    for _, (bar, val) in enumerate(zip(bars, sorted_values)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va='center', fontsize=10)
    plt.gca().invert_yaxis()
    plt.show()


def summarize_model_results(model_data, primary_metric, metrics_to_display, pipelines=None):
    data = pd.DataFrame(model_data).T
    primary_metric = primary_metric.capitalize()

    data_sorted = data.sort_values(by=primary_metric, ascending=False)

    worst_model = data_sorted.index[-1]
    best_model = data_sorted.index[0]

    styled = data_sorted[list(metrics_to_display.keys())
                         ].style.format(metrics_to_display)

    print(
        f"Worst model by {primary_metric}: {worst_model} with a score of "
        f"{data_sorted.loc[worst_model, primary_metric]:.2%}"
    )
    print(
        f"Best model by {primary_metric}: {best_model} with a score of "
        f"{data_sorted.loc[best_model, primary_metric]:.2%}"
    )
    display(styled)

    return best_model, pipelines[best_model]
