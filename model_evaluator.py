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

def compute_classification_metrics(y_test, y_pred, y_proba):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    cnf_mtrx = confusion_matrix(y_test, y_pred)


def print_classification_report():
    pass


#-------------------
# Plotting functions
#-------------------

def plot_roc_curve():
    pass

def plot_precision_recal_curve():
    pass

def plot_confusion_matrix():
    pass

def plot_threshold_sweep():
    pass


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