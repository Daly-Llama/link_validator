import joblib
import shap
from pathlib import Path
from matplotlib import pyplot as plt
from text_classifier import (
    load_link_data, load_article_data, load_incident_data,
    merge_dataframes, build_features
)

#---------------------
# Set global variables
#---------------------

MODEL_PATH = Path("model.pkl")
N_SAMPLES = 200
EXPLAINER = None


#------------------------------
# SHAP explainability functions
#------------------------------

def get_explainer(model, X_background):

    global EXPLAINER

    if EXPLAINER is None:

        def predict_positive(X):
            return model.predict_proba(X)[:, 1]

        EXPLAINER = shap.KernelExplainer(predict_positive, X_background)

    return EXPLAINER

def explain_model(explainer, X_sample, feature_names):
    """
    #     Creates SHAP plots to summarize the model
    #
    #     :param model: model from text_classifier.py
    #     :param X_sample: shap.KernelExplainer object
    #     :param feature_names: list containing names of the features
    #     """

    # Get the SHAP values
    shap_values = explainer.shap_values(X_sample, nsamples=N_SAMPLES)

    # Summary plot
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
    plt.show()

    # Bar plot
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar")
    plt.show()


def explain_local(explainer, feature_vector, feature_names):
    """
    Creates a SHAP explanation for a single prediction

    :param explainer: shap.KernelExplainer object
    :param feature_vector: feature array
    :param feature_names: list of strings with feature names
    :return:
    """

    shap_values = explainer.shap_values(feature_vector.reshape(1, -1), nsamples=200)

    shap.force_plot(
        explainer.expected_value,
        shap_values,
        feature_vector,
        feature_names=feature_names,
        matplotlib=True
    )
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

    # Get the SHAP Explainer
    background = shap.sample(X, 50)
    explainer = get_explainer(model, background)

    # Get a sample of the features for SHAP
    X_sample = shap.sample(X, N_SAMPLES)

    # Run Global Explanation
    explain_model(explainer, X_sample, feature_columns)

    # Get an explanation for a single row
    row = X[0]
    explain_local(explainer, row, feature_columns)


if __name__ == "__main__":
    main()