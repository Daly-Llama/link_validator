import json
import joblib
import pandas as pd
from pathlib import Path
from text_embedder import extract_article_text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


#---------------------
# Set global variables
#---------------------

# Files and directories
LINK_DATA_DIR = Path(r'links.jsonl')
INCIDENT_DATA_DIR = Path('raw_data/inc_json')
ARTICLE_DATA_DIR = Path('raw_data/json')

# Select the final features to use
FEATURE_COLUMNS = [
    'similarity',
    'distance',
    'text_length_ratio',
    'category_match',
    'subcategory_match'
]

# Training parameters
TEST_SIZE = .2
RANDOM_STATE = 48
STRATIFY = True

# Logistic Regression hyperparameters
C = 1.0
PENALTY = 'l2'
MAX_ITER = 1000


#---------------------
#Load in all datasets
#---------------------

def load_link_data():
    """
    Reads in the links.jsonl file and returns it as a Pandas Dataframe
    """

    links = []

    with LINK_DATA_DIR.open('r', encoding='utf-8') as file:

        for line in file:
            links.append(json.loads(line))

    return pd.DataFrame(links)


def load_article_data():
    """
    Returns the required article data in a Pandas Dataframe
    """
    articles = []

    for file in ARTICLE_DATA_DIR.glob("*.json"):
        article_json = json.loads(file.read_text(encoding='utf-8'))
        article_id = article_json['id']
        article_text = extract_article_text(article_json)
        article_category = article_json['metadata']['category']
        article_subcategory = article_json['metadata']['subcategory']

        articles.append({
            "article_id": article_id,
            "article_category": article_category,
            "article_subcategory": article_subcategory,
            "article_text": article_text
        })

    return pd.DataFrame(articles)

def load_incident_data():
    """
    Returns the required Incident fields as a Pandas Dataframe
    """

    incidents = []

    for file in INCIDENT_DATA_DIR.glob("*.json"):
        incident_json = json.loads(file.read_text(encoding='utf-8'))
        incident_id = incident_json['number']
        incident_text = (
            f"{incident_json['short_description']}\n"
            f"{incident_json['description']}"
        )

        incidents.append({
            'incident_id': incident_id,
            'incident_category': incident_json['category'],
            'incident_subcategory': incident_json['subcategory'],
            'incident_text': incident_text
        })

    return pd.DataFrame(incidents)


def merge_dataframes(link_df, inc_df, kb_df):
    """Create a single dataframe that is merged with all data."""

    return (
        link_df
            .merge(inc_df, how='left', on='incident_id')
            .merge(kb_df, how='left', on='article_id')
    )


#--------------------------------------
# Add features and build the classifier
#--------------------------------------

def build_features(df):
    """
    Creates new feature columns for the dataframe. Returns the target
    and features as numpy arrays.
    """

    # Features related to the length of textual fields
    df['incident_length'] = df['incident_text'].apply(lambda x: len(x))
    df['article_length'] = df['article_text'].apply(lambda x: len(x))
    df['text_length_ratio'] = df['incident_length'] / df['article_length'].replace(0, 1)

    # Features related to matching categories/subcategories
    df['category_match'] = (df['incident_category'] == df['article_category']).astype(int)
    df['subcategory_match'] = (df['incident_subcategory'] == df['article_subcategory']).astype(int)

    # Other features
    df['distance'] = 1 - df['similarity']

    features = df[FEATURE_COLUMNS].to_numpy()

    # Select the target
    target = df['label'].to_numpy()

    return features, target


def main():

    # Load and merge the data
    links = load_link_data()
    incidents = load_incident_data()
    articles = load_article_data()
    df = merge_dataframes(links, incidents, articles)

    # Create features and extract target
    X, y = build_features(df)

    # split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if STRATIFY else None
    )

    # Train the model
    model = LogisticRegression(
        C=C,
        max_iter=MAX_ITER,
        penalty=PENALTY,
        solver = "lbfgs"
    )
    model.fit(X_train, y_train)

    # Evaluate the models performance against the testing data
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("\n", confusion_matrix(y_test, y_pred))

    # Save the model
    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
        },
        'model.pkl'
    )


if __name__ == "__main__":
    main()