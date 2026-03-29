import json
import random
from openai import OpenAI
from pathlib import Path

# ----------------------
# Set global variables
# ----------------------
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.0
OPENAI_TOP_P = 1.0
OPENAI_API_KEY = json.loads(Path(r'OPENAI.json').read_text())['openai']

SHORT_DESCRIPTION_PROMPT_DIR = "prompts/short_description_1.txt"
DESCRIPTION_PROMPT_DIR = "prompts/description_1.txt"
COMBINED_PROMPT_DIR = "prompts/incident_prompt.txt"

BASE_INC_NUMBER = 1000


# ----------------------
# Establish OpenAI client
# ----------------------

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

# ----------------------
# Read json KB article files
# ----------------------

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


def unpack_kb_content(article_json):
    """
    Unpacks the article JSON and returns a dictionary of the fields needed for
    generating synthetic incidents
    """

    article_id = article_json['id']
    category = article_json['metadata']['category']
    subcategory = article_json['metadata']['subcategory']
    article_type = 'Question' if 'Question' in article_json['body'] else 'Issue/Question'
    title = article_json['metadata']['title']
    issue_question = article_json['body'][article_type]
    environment = '' if article_type == 'Question' else article_json['body']['Environment']

    combined_content = (
        f"Title: {title}\n\n"
        f"Environment: {environment}\n\n"
        f"Issue/Question:\n{issue_question}"
    )

    return {
        "article_id": article_id,
        "category": category,
        "subcategory": subcategory,
        "article_type": article_type,
        "content": combined_content
    }


# ----------------------
# Situational Modifiers for Prompts
# ----------------------

def random_situational_mod(json_file):
    """
    Returns a randomized modifier from a .json file in the
    situational_modifiers directory.
    """

    # Read in the JSON and extract all the items
    data = json.loads(Path(json_file).read_text())
    items = data['items']

    # Generate a random number between 0 and 1
    randomized_number = random.random()
    cumulative_total = 0.0

    # Iterate until the cumulative_total is higher than randomize_number
    for key, value in items.items():
        cumulative_total += value['weight']
        if randomized_number < cumulative_total:
            return key, value

    # As a fallback, return the last item
    last_key = next(reversed(items))
    return last_key, items[last_key]


def build_situational_modifier():
    """
    Generates situational modifier text for the main short_description and
    description prompts.

    :return: a Tuple with the Modifiers Used and the 2 modifier text strings
    """

    # Generate random modifiers
    fldr = r'prompts/situational_modifiers/'
    language_mod = random_situational_mod(fldr+'native_language.json')
    proficiency_mod = random_situational_mod(fldr+'technical_proficiency.json')
    spelling_mod = random_situational_mod(fldr+'spelling_grammar.json')
    contextual_noise_mod = random_situational_mod(fldr+'contextual_noise.json')
    detail_level_mod = random_situational_mod(fldr+'detail_level.json')
    structure_mod = random_situational_mod(fldr + 'structure_style.json')
    writing_style_mod = random_situational_mod(fldr+'writing_style.json')

    # Unpack the modifier text
    language = language_mod[1]['value']
    proficiency = proficiency_mod[1]['value']
    spelling = spelling_mod[1]['value']
    contextual_noise = contextual_noise_mod[1]['value']
    detail_level = detail_level_mod[1]['value']
    structure = structure_mod[1]['value']
    writing_style = writing_style_mod[1]['value']

    # Create the full modifier text
    situtational_modifier = (
        "When creating the incident text, act as someone who speaks "
        f"{language} as their first language.\n"
        f"You should write the description in {structure} and use "
        f"{writing_style} tone. The writing should have {detail_level} and "
        f"show {proficiency}. There should be {spelling} and some {contextual_noise}."
    )

    # Log which modifiers were used for tracking and reproducibility
    modifiers_used = {
        "native_language": language_mod[0],
        "technical_proficiency": proficiency_mod[0],
        "spelling_grammar": spelling_mod[0],
        "contextual_noise": contextual_noise_mod[0],
        "detail_level": detail_level_mod[0],
        "structure_style": structure_mod[0],
        "writing_style": writing_style_mod[0]
    }

    return modifiers_used, situtational_modifier


# ----------------------
# Main Prompts
# ----------------------

def build_short_desc_prompt(kb_text, modifier_text):
    """Builds the full prompt to generate an incident short_description"""

    # Insert the kb_text and situational modifiers into the prompt template
    prompt = Path(SHORT_DESCRIPTION_PROMPT_DIR).read_text()
    prompt = prompt.replace("[ARTICLE_TEXT]", kb_text)
    prompt = prompt.replace("[SITUATIONAL_MODIFIERS]", modifier_text)

    return prompt


def build_description_prompt(kb_text, modifier_text):
    """Builds the full prompt to generate an incident description"""

    prompt = Path(DESCRIPTION_PROMPT_DIR).read_text()
    prompt = prompt.replace("[ARTICLE_TEXT]", kb_text)
    prompt = prompt.replace("[SITUATIONAL_MODIFIERS]", modifier_text)

    return prompt


def build_combined_prompt(kb_text, modifier_text):
    """Builds the full prompt to generate an incident short_description"""

    # Insert the kb_text and situational modifiers into the prompt template
    prompt = Path(COMBINED_PROMPT_DIR).read_text()
    prompt = prompt.replace("[ARTICLE_TEXT]", kb_text)
    prompt = prompt.replace("[SITUATIONAL_MODIFIERS]", modifier_text)

    return prompt

# -------------------------
# INCIDENT GENERATION (LEAN SCHEMA)
# -------------------------

def generate_incident(inc_number, article_json):
    """
    Generate synthetic incidents using the lean schema.

    """


    incidents = []


def test_prompt(kb_file):
    file = f'raw_data/json/{kb_file}'

    j = json.loads(Path(file).read_text())
    content = unpack_kb_content(j)

    modifier = build_situational_modifier()

    prompt = build_combined_prompt(content['content'], modifier[1])

    return prompt





