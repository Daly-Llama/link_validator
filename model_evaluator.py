import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, accuracy_score
)
from text_classifier import (
    load_link_data, load_article_data, load_incident_data,
    merge_dataframes, build_features
)

#---------------------
# Set global variables
#---------------------

MODEL_PATH = Path("model.pkl")


#----------------
# Summary results
#----------------

def calculate_classification_metrics(y_test, y_pred, y_proba):
    """
    Returns the metrics for the classifier's performance.
    """

    classification_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return classification_metrics


def print_classification_report(metrics):
    """Prints the evaluation metrics for the model"""

    print(f"{'Evaluation Metrics':^24}")
    print(f"{'Accuracy:':<15}{metrics['accuracy']:>9.4f}")
    print(f"{'Precision:':<15}{metrics['precision']:>9.4f}")
    print(f"{'Recall:':<15}{metrics['recall']:>9.4f}")
    print(f"{'F1 Score:':<15}{metrics['f1_score']:>9.4f}")
    print(f"{'ROC-AUC:':<15}{metrics['roc_auc']:>9.4f}")

#-------------------
# Plotting functions
#-------------------

def plot_roc_curve(y_test, y_proba):
    """Creates an ROC-AUC plot"""

    # Calculate Receiver Operating Characteristic (ROC) and Area Under Curve (AUC)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_proba)
    area_under_curve = roc_auc_score(y_test, y_proba)

    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0,1], [0,1])
    plt.title("ROC-AUC Curve")
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.show()

def plot_precision_recal_curve(y_test, y_proba):
    """Plots the models recall vs. precision"""

    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision/Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    """Creates a confusion matrix plot"""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    plt.imshow(cm, cmap="Oranges")
    plt.colorbar()

    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    # Add lables
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_threshold_sweep(y_test, y_proba):
    thresholds = np.linspace(0, 1, 101)
    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_test, preds))
        recalls.append(recall_score(y_test, preds))
        f1s.append(f1_score(y_test, preds))

    # Create and display the plots
    plt.figure()
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1s, label="F1 Score")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sweep")
    plt.legend()
    plt.show()


#------------------
# __main__ function
#------------------

def main():

    # Load the data
    links = load_link_data()
    incidents = load_incident_data()
    articles = load_article_data()
    df = merge_dataframes(links, incidents, articles)

    # Build the features
    X, y = build_features(df)

    # Load and unpack the model
    bundle = joblib.load(MODEL_PATH)
    model = bundle['model']
    feature_columns = bundle['features']

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >=.5).astype(int)

    classification_metrics = calculate_classification_metrics(y, y_pred, y_proba)
    print_classification_report(classification_metrics)

    plot_roc_curve(y, y_proba)
    plot_precision_recal_curve(y, y_proba)
    plot_confusion_matrix(y, y_pred)
    plot_threshold_sweep(y, y_proba)


if __name__ == "__main__":
    main()