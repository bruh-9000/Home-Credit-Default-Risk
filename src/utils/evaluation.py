import warnings
import joblib
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from matplotlib import pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

warnings.filterwarnings('ignore')
np.random.seed(42)

from src.utils.config import (
    SAVE_PATH,
    label,
    value_mappings,
    model_configs
)

from modeling import train_pipeline, predict_pipeline



def evaluate_pipeline(name, data):
    _, X_train, y_train, X_test, y_test = data

    label_mapping = value_mappings.get(label)
    if label_mapping:
        y_train = y_train.map(label_mapping)
        if y_test is not None:
            y_test = y_test.map(label_mapping)

    has_test_labels = y_test is not None and not pd.isna(y_test).all()
    pipe = train_pipeline(name, X_train, y_train)

    if has_test_labels:
        y_preds, y_probs = predict_pipeline(pipe, X_test)
        print(f"\n{name} - TEST METRICS:")
        output_metrics = get_analysis(y_test, y_preds, y_probs)

        if not model_configs.get(name, {}).get('baseline', False):
            generate_graphs(pipe, X_train, y_test, y_preds, y_probs)
    else:
        print(f"\n{name} - SKIPPING TEST (no labels)")

    if not model_configs.get(name, {}).get('baseline', False):
        joblib.dump(pipe, SAVE_PATH / f"{name}_pipeline.pkl")

    return pipe, output_metrics



def get_analysis(y_var, y_preds, y_probs=None):
    tn, fp, fn, tp = confusion_matrix(y_var, y_preds).ravel()
    accuracy = accuracy_score(y_var, y_preds)
    f1 = f1_score(y_var, y_preds)
    recall = recall_score(y_var, y_preds)
    precision = precision_score(y_var, y_preds)
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    # ROC AUC only if probabilities are passed
    roc_auc = roc_auc_score(y_var, y_probs) if y_probs is not None else None

    report = f"""
Accuracy: {accuracy:.2%}
F1 Score: {f1:.2%}
Precision: {precision:.2%}
True Positive Rate (Recall): {recall:.2%}
True Negative Rate (Specificity): {specificity:.2%}
False Positive Rate: {fpr:.2%}
False Negative Rate: {fnr:.2%}
{'ROC AUC: {:.2%}'.format(roc_auc) if roc_auc is not None else ''}
"""
    
    values = {
        'Accuracy': accuracy,
        'F1': f1,
        'Precision': precision,
        'Recall (TPR)': recall,
        'Specificity (TNR)': specificity,
        'FPR': fpr,
        'FNR': fnr
    }

    if roc_auc is not None:
        values['Roc_auc'] = roc_auc

    print(report)

    return values



def generate_graphs(model, X_train, y_test, y_pred, y_probs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_confusion_matrix(y_test, y_pred, ax=axes[0])
    plot_roc_curve(y_test, y_probs, ax=axes[1])
    plot_pr_curve(y_test, y_probs, ax=axes[2])

    plt.tight_layout()
    plt.show()

    plot_shap_summary(model, X_train)



def plot_confusion_matrix(y_test, y_pred, labels=['Negative', 'Positive'], ax=None):
    cm = confusion_matrix(y_test, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)



def plot_roc_curve(y_test, y_probs, ax=None):
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    ax.grid(True)



def plot_pr_curve(y_test, y_probs, ax=None):
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    avg_precision = average_precision_score(y_test, y_probs)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, label=f"Avg Precision = {avg_precision:.4f}")
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(True)



def plot_shap_summary(model, X_train, num_samples=20):
    features = X_train.columns
    X_sample = pd.DataFrame(X_train[:num_samples], columns=features)

    explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap.summary_plot(shap_values, X_sample, feature_names=features, plot_type='bar', plot_size=[8, 4])



def summarize_model_results(model_data, primary_metric, metrics_to_display, pipelines=None):
    df = pd.DataFrame(model_data).T
    primary_metric = primary_metric.capitalize()

    df_sorted = df.sort_values(by=primary_metric, ascending=False)

    worst_model = df_sorted.index[-1]
    best_model = df_sorted.index[0]

    styled = df_sorted[list(metrics_to_display.keys())].style.format(metrics_to_display)

    display(Markdown(f"**Worst model by {primary_metric}:** {worst_model} with a score of {df_sorted.loc[worst_model, primary_metric]:.2%}"))
    display(Markdown(f"**Best model by {primary_metric}:** {best_model} with a score of {df_sorted.loc[best_model, primary_metric]:.2%}"))
    display(styled)

    return best_model, pipelines[best_model]