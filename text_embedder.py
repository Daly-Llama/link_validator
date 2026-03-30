import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ----------------------
# Environment Variables
# ----------------------
ARTICLE_DIR = Path("raw_data/json")
INCIDENT_DIR = Path("raw_data/inc_json")

EMBED_SAVE_DIR = Path("embeddings")
EMBED_SAVE_DIR.mkdir(parents=True, exist_ok=True)

ARTICLE_EMBED_FILE = EMBED_SAVE_DIR / "articles.jsonl"
INCIDENT_EMBED_FILE = EMBED_SAVE_DIR / "incidents.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ----------------------
# Loading the model
# ----------------------

def load_sentence_transformer():
    return SentenceTransformer(MODEL_NAME)


# --------------------------
# Embedding lookup & writing
# --------------------------

def get_pre_existing_embeddings(embedding_path, field):
    """
    Get the IDs of any embeddings that have already been added
    to the specified embedding_path
    """

    # Handle when embedding file has not been created yet
    if not embedding_path.exists():
        return set()

    # Read each individual embedding record
    existing_ids = set()
    with embedding_path.open('r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            existing_ids.add(item[field])

    return existing_ids


def write_jsonl_line(path, record):
    """
    Writes a new JSONL line to the file at the specified path
    for the record
    """

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ----------------------
# Text extracting
# ----------------------

def extract_article_text(article_json):
    """
    Extracts the body fields of an article and returns it as a
    formatted string for use in embedding.
    """

    title = article_json['metadata']['title']
    article_text = f"Title: {title}"

    # Iterate through the possible body fields and concat to the output text
    field_order = [
        'Issue/Question', 'Issue', 'Question',
        'Environment', 'Cause', 'Answer', 'Resolution'
    ]
    for field in field_order:
        try:
            field_text = article_json['body'][field]
            article_text += f"\n\n{field}\n{field_text}"
        except KeyError:
            continue

    return article_text.strip()



def extract_incident_text(incident_json):
    """
    Returns the extracted text of an incident's short description
    and description fields for use in embedding.
    """

    # Attempt to extract the short description
    try:
        short_desc = incident_json['short_description']
        incident_text = f"Short Description: {short_desc}"
    except KeyError:
        incident_text = ""

    # Attempt to extract the description
    try:
        description = incident_json['description']
        incident_text += f"\n\nDescription:\n{description}"
    except KeyError:
        incident_text += ""

    if incident_text == "":
        return None
    else:
        return incident_text


# ----------------------
# Get embeddings
# ----------------------

def embed_new_articles(model):
    """
    Obtains the embeddings for any new KCS articles that haven't already
    been added to the ARTICLE_DIR and appends them to the ARTICLE_EMBED_FILE
    """

    pre_existing_embeddings = get_pre_existing_embeddings(ARTICLE_EMBED_FILE, "article_id")

    for file in ARTICLE_DIR.glob("*.json"):

        article_json = json.loads(file.read_text(encoding="utf-8"))
        article_id = article_json['id']

        # Skip if the article is already embedded
        if article_id in pre_existing_embeddings:
            continue

        article_text = extract_article_text(article_json)
        vector = model.encode(article_text).tolist()

        record = {
            'article_id': article_id,
            'embedding': vector
        }

        write_jsonl_line(ARTICLE_EMBED_FILE, record)

        if article_id.endswith("00"):
            print(f"Completed embeddings for {article_id}")


def embed_new_incidents(model):
    """
    Obtains the embeddings for any new KCS incidents that haven't already
    been added to the INCIDENT_DIR and appends them to the INCIDENT_EMBED_FILE
    """

    pre_existing_embeddings = get_pre_existing_embeddings(INCIDENT_EMBED_FILE, "incident_id")

    for file in INCIDENT_DIR.glob("*.json"):

        incident_json = json.loads(file.read_text(encoding="utf-8"))
        incident_number = incident_json['number']

        # Skip if the incident is already embedded
        if incident_number in pre_existing_embeddings:
            continue

        incident_text = extract_incident_text(incident_json)

        # Skip if there isn't any text in the incident
        if incident_text is None:
            continue

        vector = model.encode(incident_text).tolist()

        record = {
            'incident_id': incident_number,
            'embedding': vector
        }

        write_jsonl_line(INCIDENT_EMBED_FILE, record)

        #
        if incident_number.endswith("00"):
            print(f"Completed embeddings for {incident_number}")

def main():
    model = load_sentence_transformer()
    embed_new_articles(model)
    print('Article Embeddings complete.')
    embed_new_incidents(model)
    print('Incident Embeddings complete.')


if __name__ == "__main__":
    main()