import json
import pandas as pd
from openai import OpenAI
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.0
OPENAI_TOP_P = 1.0

BASE_INC_NUMBER = 1000
PROMPT_DIR = "prompts"


client = OpenAI()


def generate_text(prompt):
    """
    Sends a deterministic prompt to OpenAI and returns the generated text.
    """
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        top_p=OPENAI_TOP_P,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content.strip()


def load_json_files(directory_path):
    """
    A generator function that iterates through each JSON file in
    the directory path and returns its article ID and contents

    :param directory_path: string representing the directory path
    :return: a JSON object representing the article
    """

    directory = Path(directory_path)

    for file in directory.glob("*.html"):
        article_id = file.stem
        content = file.read_text(encoding='utf-8', errors='ignore')
        json_content = json.loads(content)
        yield json_content


def build_short_desc_prompt(issue_text, idx):
    """Build a short_description prompt using deterministic rotation."""
    pass


def build_description_prompt(issue_text, idx):
    """Build a description prompt using deterministic rotation."""
    pass

# -------------------------
# INCIDENT GENERATION (LEAN SCHEMA)
# -------------------------

def generate_incident(inc_number, article_json):
    """
    Generate synthetic incidents using the lean schema.

    """

    article_id = article_json['id']
    category = article_json['metadata']['category']
    subcategory = article_json['metadata']['subcategory']

    article_type = 'Question' if 'Question' in article_json['body'] else 'Issue/Question'
    issue_text = article_json['body'][article_type]


    incidents = []