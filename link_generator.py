import json
import numpy as np
from pathlib import Path


#---------------------
# Set global variables
#---------------------

ARTICLE_EMBED_FILE = Path(r'embeddings/articles.jsonl')
INCIDENT_EMBED_FILE = Path(r'embeddings/incidents.jsonl')
INCIDENT_JSON_DIR = Path(r'raw_data/inc_json')

OUTPUT_LINK_FILE = Path("links.jsonl")

TOP_K_LINK_NUM = 5


#--------------------------------
# Read in the Embeddings and Data
#--------------------------------

def get_article_embeddings():
    """
    Loads in the ARTICLE_EMBED_FILE and returns a list of all its article_ids
    and a numpy array with its embeddings.
    """
    article_ids = []
    vectors = []

    with ARTICLE_EMBED_FILE.open('r', encoding='utf-8') as f:

        for line in f:
            article_json = json.loads(line)
            article_id = article_json['article_id']
            vector = article_json['embedding']

            article_ids.append(article_id)
            vectors.append(vector)

    return article_ids, np.array(vectors, dtype=np.float32)


def get_incident_embeddings():
    """
    Loads in the INCIDENT_EMBED_FILE and returns a list of all its incident_ids
    and a numpy array with its embeddings.
    """
    incident_ids = []
    vectors = []

    with INCIDENT_EMBED_FILE.open('r', encoding='utf-8') as f:

        for line in f:
            incident_json = json.loads(line)
            incident_id = incident_json['incident_id']
            vector = incident_json['embedding']

            incident_ids.append(incident_id)
            vectors.append(vector)

    return incident_ids, np.array(vectors, dtype=np.float32)


def get_incident_metadata(incident_ids):
    """
    Get the category, subcategory, and correct_kb for all incidents in the
    incident_ids list.
    """

    metadata = {}

    for incident_id in incident_ids:
        file = INCIDENT_JSON_DIR / f"{incident_id}.json"
        incident_json = json.loads(file.read_text(encoding='utf-8'))

        metadata[incident_id] = {
            'incident_id': incident_id,
            'category': incident_json['category'],
            'subcategory': incident_json['subcategory'],
            'correct_kb': incident_json['correct_kb']
        }

    return metadata


#-----------------------------
# Similarity and Top Matches
#-----------------------------

def compute_cosine_similarity(vector1, vector2):
    """ Returns the Cosine similarity between 2 vectors."""

    dot_product = np.dot(vector1, vector2)
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    return float(dot_product / denominator)


def get_top_k_articles(inc_vectors, kb_vectors, article_ids, k=TOP_K_LINK_NUM):
    """
    Computes similarity between one incident and all articles.
    Returns a list of (article_id, similarity) sorted descending.
    """
    normalized_kbs = kb_vectors / np.linalg.norm(kb_vectors, axis=1, keepdims=True)
    normalized_incs = inc_vectors / np.linalg.norm(inc_vectors)

    similarities = normalized_kbs @ normalized_incs

    # Get top-k indices
    top_k_indices = np.argsort(similarities)[::-1][:k]

    return [(article_ids[i], float(similarities[i])) for i in top_k_indices]


#-----------------------------
# Generate the linking data
#-----------------------------

def generate_link_candidates(inc_ids, inc_vectors, kb_ids, kb_vectors, k=TOP_K_LINK_NUM):
    # Returns the 5 most similar articles as candidates for incorrect links.

    link_candidates = {}

    for index, inc_id in enumerate(inc_ids):
        inc_vector = inc_vectors[index]
        top_k_kbs = get_top_k_articles(inc_vector, kb_vectors, kb_ids, k=k)
        link_candidates[inc_id] = top_k_kbs

    return link_candidates


#-----------------------------
# Create and save the data
#-----------------------------

def create_links_dataset(candidates, inc_metadata):

    links = []

    for inc_id, similar_kbs in candidates.items():
        metadata = inc_metadata.get(inc_id, {})
        correct_kb = metadata.get("correct_kb")

        for kb_id, similarity in similar_kbs:
            label = 1 if kb_id == correct_kb else 0

            links.append({
                'incident_id': inc_id,
                'article_id': kb_id,
                'similarity': similarity,
                'label': label
            })

    return links


def write_to_jsonl(link_records, output_path=OUTPUT_LINK_FILE):

    with output_path.open('w', encoding='utf-8') as f:
        for link in link_records:
            f.write(json.dumps(link) + '\n')


def main():

    article_ids, article_vctrs = get_article_embeddings()
    incident_ids, incident_vctrs = get_incident_embeddings()
    incident_metadata = get_incident_metadata(incident_ids)
    candidates = generate_link_candidates(
        incident_ids, incident_vctrs, article_ids, article_vctrs
    )
    link_dataset = create_links_dataset(candidates, incident_metadata)
    write_to_jsonl(link_dataset)

if __name__ == "__main__":
    main()