import json
import random
import re
from openai import OpenAI
from pathlib import Path

# ----------------------
# Set global variables
# ----------------------
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.0
OPENAI_TOP_P = 1.0
OPENAI_API_KEY = json.loads(Path(r'OPENAI.json').read_text())['openai']

PROMPT_DIR = "prompts/incident_prompt.txt"
JSON_FILE_DIRECTORY = r"raw_data/json"
INC_SAVE_DIRECTORY = r"raw_data/inc_json"

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

    for file in directory.glob("*.json"):
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

    title = article_json['metadata']['title']

    # Get the article-type
    article_type = None
    if 'Question' in article_json['body']:
        article_type = 'Question'
    elif 'Issue/Question' in article_json['body']:
        article_type = 'Issue/Question'
    elif 'Issue' in article_json['body']:
        article_type = 'Issue'

    issue_question = article_json['body'][article_type]

    # Get the environment
    try:
        environment = article_json['body']['Environment']
    except KeyError:
        environment = ''

    combined_content = (
        f"Title: {title}\n\n"
        f"Environment: \n{environment}\n\n"
        f"Issue/Question:\n{issue_question}"
    )

    return {
        "article_id": article_id,
        "category": category,
        "subcategory": subcategory,
        "article_type": article_type,
        "content": combined_content
    }


# ---------------------------------
# Situational Modifiers for Prompts
# ---------------------------------

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
    entry_method_mod = random_situational_mod(fldr+'entry_method.json')

    # Unpack the modifier text
    language = language_mod[1]['value']
    proficiency = proficiency_mod[1]['value']
    spelling = spelling_mod[1]['value']
    contextual_noise = contextual_noise_mod[1]['value']
    detail_level = detail_level_mod[1]['value']
    structure = structure_mod[1]['value']
    writing_style = writing_style_mod[1]['value']
    entry_method = entry_method_mod[1]['value']

    # Create the full modifier text
    situational_modifier = (
        f"Assume that the ticket was created {entry_method} and that the person who "
        f"created it speaks {language} as their first language.\n"
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
        "writing_style": writing_style_mod[0],
        "entry_method": entry_method_mod[0]
    }

    return modifiers_used, situational_modifier


# -------------------------
# OpenAI Prompt
# -------------------------

def build_full_prompt(kb_text, modifier_text):
    """Builds the full prompt to generate the text of an incident"""

    # Insert the kb_text and situational modifiers into the prompt template
    prompt = Path(PROMPT_DIR).read_text()
    prompt = prompt.replace("[ARTICLE_TEXT]", kb_text)
    prompt = prompt.replace("[SITUATIONAL_MODIFIERS]", modifier_text)

    return prompt


# -------------------------
# INCIDENT GENERATION
# -------------------------

def generate_incident_text(kb_content, modifier_text):
    """
    Generate synthetic incidents using the lean schema
    """

    prompt = build_full_prompt(kb_content, modifier_text)

    response = generate_text(prompt)

    return response


def parse_incident_text(text):
    """
    Attempts to parse out the short_description and description fields of a reply
    from OpenAI and returns them as a dictionary.

    :param text: the reply text from OpenAI
    :return: dictionary with unpacked description and short_description
    """

    parsed_text = {'raw_text': text}

    # Attempt to extract the short description
    try:
        short_description = re.match(r'short_description_text:\s+(.*)', text)
        parsed_text.setdefault('short_description', short_description.group(1))
    except AttributeError:
        parsed_text.setdefault('short_description', None)

    # Attempt to extract the description
    try:
        description = re.search(r'full_description_text:\s+(.*)', text, re.DOTALL)
        parsed_text.setdefault('description', description.group(1))
    except AttributeError:
        parsed_text.setdefault('description', None)

    return parsed_text


def generate_incident_fields(inc_num, parsed_text, unpacked_content, modifier):
    """
    Creates the fields for a synthetic USD support incident.

    :param inc_num: the identifier of the incident to create
    :param parsed_text: output of parse_incident_text()
    :param unpacked_content: output of unpack_kb_content()
    :param modifier: output of build_situational_modifier()
    :return: a dictionary with the field values for the synthetic incident
    """

    incident_metadata = {
        "number": inc_num,
        "category": unpacked_content['category'],
        "subcategory": unpacked_content['subcategory'],
        "correct_kb": unpacked_content['article_id'],
        "modifier_text": modifier[1],
        "modifier_choices": modifier[0]
    }

    return {**incident_metadata, **parsed_text}


def save_incident_json(incident_json, file_path):
    """
    Saves the processed incident to a json file.

    :param incident_json: output of generate_incident_fields
    :param file_path: the path to save to
    """

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(incident_json, f, indent=2, ensure_ascii=False)


def main():

    inc_number = BASE_INC_NUMBER

    for json_content in load_json_files(JSON_FILE_DIRECTORY):

        inc_number += 1
        inc_identifier = f"INC{inc_number:07d}"

        json_directory = Path(INC_SAVE_DIRECTORY)
        json_directory.mkdir(parents=True, exist_ok=True)

        inc_path = json_directory / f"{inc_identifier}.json"

        if inc_path.exists():
            continue

        try:
            kb_content = unpack_kb_content(json_content)
        except Exception as e:
            print("\n--- ERROR PROCESSING ARTICLE ---")
            print("Article ID:", json_content.get("id"))
            print("Body keys:", list(json_content.get("body", {}).keys()))
            print("Full body:", json_content.get("body"))
            print("Error:", e)
            continue

        modifier = build_situational_modifier()

        inc_text = generate_incident_text(kb_content['content'], modifier[1])
        parsed_text = parse_incident_text(inc_text)
        inc_json = generate_incident_fields(inc_identifier, parsed_text, kb_content, modifier)

        save_incident_json(inc_json, inc_path)


if __name__ == "__main__":
    main()