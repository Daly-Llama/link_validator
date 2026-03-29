import re
import json
from bs4 import BeautifulSoup
from pathlib import Path


# GLOBAL VARIABLES
HTML_FILE_DIRECTORY = r'raw_data/html'
JSON_FILE_DIRECTORY = r'raw_data/json'


def load_html_files(directory_path):
    """
    A generator function that iterates through each HTML file in
    the directory path and returns its article ID and contents

    :param directory_path: string representing the directory path
    :return: 2D tuple containing 1. the article ID and 2. the HTML
    """

    directory = Path(directory_path)

    for file in directory.glob("*.html"):
        article_id = file.stem
        html = file.read_text(encoding='utf-8', errors='ignore')
        yield article_id, html


def clean_html_data(html_text):
    """
    Removes certain elements of the HTML that are problematic for
    parsing the contents.

    :param html_text: the text of an HTML file
    :return: the HTML text, without the problematic elements
    """

    # Remove script blocks
    html_text = re.sub(
        r'<script.*?>.*?</script>', '', html_text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Remove style blocks
    html_text = re.sub(
        r'<<style.*?>.*?</style>>', '', html_text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Remove comments
    html_text = re.sub(
        r'<!--.*?-->', '', html_text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Replace duplicate whitespaces
    html_text = re.sub(
        r'\s+', ' ', html_text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Return the final output
    return html_text.strip()


def make_soup(html_text):
    """
    Returns a BeautifulSoup object for the html_text passed as input
    """
    soup = BeautifulSoup(html_text)
    return soup


def extract_metadata(soup):
    """
    Extracts the metadata elements that are structured into a BeautifulSoup
    object and returns them as a dictionary.
    """

    metadata = {}

    h1 = soup.find('h1')
    metadata['title'] = h1.get_text(strip=True) if h1 else None

    published = soup.find("meta", {"property": "article:published_time"})
    metadata['published'] = published['content'] if published else None

    modified = soup.find("meta", {"property": "article:modified_time"})
    metadata['updated'] = modified['content'] if modified else None

    breadcrumb = soup.find('ol', class_=lambda c: c and 'breadcrumb' in c)

    if breadcrumb:
        hierarchy = breadcrumb.find_all('li')
        hierarchy_list = [i.get_text(strip=True) for i in hierarchy]
        metadata['breadcrumb'] = hierarchy_list

        # Set the main category
        kb_category = hierarchy[1].get_text(strip=True)
        metadata['category'] = kb_category

        # Extract subcategory
        kb_subcategory = hierarchy[-2].get_text(strip=True)
        metadata['subcategory'] = kb_subcategory

    else:
        metadata['breadcrumb'] = None
        metadata['category'] = None
        metadata['subcategory'] = None

    return metadata


def extract_article_body(soup):
    """
    Extracts the headers, body and ITS STAFF ONLY blocks of the article.

    :param soup: a BeautifulSoup object for a USD knowledge article
    :return: a dictionary. Key=header, Value=text in that heading
    """

    # Fields to assist with the loop and collecting the final output
    content = {}
    current_header = None
    current_section_text = []

    # Extract the body of the article
    body = soup.find('div', id='ctl00_ctl00_cpContent_cpContent_divBody')
    if not body:
        return content

    # Extract ITS STAFF ONLY block
    staff_marker = body.find(string=lambda s: s and "ITS STAFF ONLY" in s.upper())
    staff_only_text = None

    # Remove the STAFF ONLY block
    if staff_marker:
        staff_block = staff_marker.find_parent("div")
        staff_only_text = staff_block.get_text("\n", strip=True)
        staff_block.decompose()  # remove from DOM so it doesn't contaminate sections

    # Store staff-only content
    if staff_only_text:
        content["staff_only"] = staff_only_text

    # Extract H2 sections from the body
    for element in body.children:

        # Detect new section headings
        if element.name == "h2":

            # Save previous section
            if current_header:
                content[current_header] = "\n".join(current_section_text).strip()
                current_section_text = []

            current_header = element.get_text(strip=True)

        # Extract section content
        elif current_header:
            text = element.get_text("\n", strip=True)
            if text:
                current_section_text.append(text)

    # Save last section
    if current_header:
        content[current_header] = "\n".join(current_section_text).strip()

    return content


def get_article_type(article_content):
    """
    Determines the type of article (Q&A, Issue, etc.)

    :param article_content: dictionary from the extract_article_body function
    :return: a string representing the article type
    """

    headers = {header.lower() for header in article_content.keys()}

    # Check for Q&A type article
    if 'answer' in headers:
        return {'type': 'qa'}

    elif 'issue/question' in headers:
        return {'type': 'issue'}

    else:
        return {'type': 'other'}


def extract_tags(soup):
    """
    Extracts the text of any tags in the article.

    :param soup: a BeautifulSoup object for a USD knowledge article
    :return: a dictionary. Key='type', Value=string of any tags, or None
    """

    tags_element = soup.find('div', id='ctl00_ctl00_cpContent_cpContent_divTags')
    if not tags_element:
        return {'tags': None}

    hyperlink_elements = tags_element.find_all('a')
    if not hyperlink_elements:
        return {'tags': None}

    else:
        hyperlink_text = [a.get_text(strip=True) for a in hyperlink_elements]
        return {'tags': hyperlink_text}


def extract_article_reviews(soup):
    """
    Extracts the number of reviews and the review score from the article.

    :param soup: a BeautifulSoup object representing a USD knowledge article
    :return: a dictionary with 2 key-value pairs for the number of reviews
            and the overall score of the reviews
    """

    # Check for a reviews element
    reviews_element = soup.find('div', id='ctl00_ctl00_cpContent_cpContent_UpdatePanel1')
    if not reviews_element:
        return {'number_reviews': None, 'review_score': None}

    # Extract the text and numbers
    reviews_text = reviews_element.get_text(strip=True)
    number_reviews = re.findall(r'(\d+) review', reviews_text)
    review_score = re.findall(r'(\d+)%', reviews_text)

    # Handle missing reviews or scores
    reviews = None if len(number_reviews) == 0 else int(number_reviews[0])
    score = None if len(review_score) == 0 else int(review_score[0])

    return {'number_reviews': reviews, 'review_score': score}


def create_article_object(article_id, metadata, body, article_type, tags, reviews):
    """
    Creates the final object containing all article contents

    :param article_id: the ID of the article
    :param metadata: output of extract_metadata()
    :param body: output of extract_article_body()
    :param article_type: output of get_article_type()
    :param tags: output of extract_tags()
    :param reviews: output of extract_reviews()
    :return: a nested dictionary with all input parameters
    """

    return {
        'id': article_id,
        'article_type': article_type['type'],
        'metadata': metadata,
        'body': body,
        'tags': tags.get('tags'),
        'reviews': reviews
    }


def save_article_json(article_id, article_obj, json_dir=JSON_FILE_DIRECTORY):
    """
    Saves the processed article to a json file.

    :param article_id: the ID for the article
    :param article_obj: output of create_article_object
    :param json_dir: string to the directory to save to
    """
    # Get the directory and filepath
    save_directory = Path(json_dir)
    save_directory.mkdir(parents=True, exist_ok=True)
    filepath = save_directory / f"{article_id}.json"

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(article_obj, f, indent=2, ensure_ascii=False)



def main():

    for article_id, html in load_html_files(HTML_FILE_DIRECTORY):
        cleaned_html = clean_html_data(html)
        soup = make_soup(cleaned_html)
        metadata = extract_metadata(soup)
        body = extract_article_body(soup)
        article_type = get_article_type(body)
        tags = extract_tags(soup)
        reviews = extract_article_reviews(soup)
        article_object = create_article_object(
            article_id, metadata, body, article_type, tags, reviews
        )
        save_article_json(article_id, article_object)


if __name__ == '__main__':
    main()