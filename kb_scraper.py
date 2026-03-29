import requests
import random
import time
from bs4 import BeautifulSoup
from pathlib import Path


# Set the global variables
BASE_KB_URL = r"https://td.usd.edu/TDClient/33/Portal/KB/ArticleDet?ID="
HTML_DIR = Path("raw_data/html")
LOG_FILE = Path("raw_data/scrape_log.csv")
START_ON_ID = 1
END_ON_ID = 10200

# Create directories for the scraped data
HTML_DIR.mkdir(parents=True, exist_ok=True)


def get_article(kb_id):
    """
    Performs a GET request for the KB article ID, and handles the response
    according to its status.

    :param kb_id: integer representing the ID number of a KB article
    :return: A tuple, where the 1st value contains the request object
    (if the request was valid) and the 2nd value is the request status.
    """

    kb_url = f"{BASE_KB_URL}{kb_id}"

    attempts = 0
    while attempts < 2:
        try:
            r = requests.get(kb_url, timeout=10)

            # Status 404 = invalid page, no retry needed
            if r.status_code == 404:
                log_result(f'KB{kb_id:05d}', 404)
                return None, 404

            # Status 200 = valid page, return page
            if r.status_code == 200:
                return r, 200

            # Status 301/302 = restricted article, skip
            if r.status_code in (301, 302):
                log_result(f'KB{kb_id:05d}', r.status_code)
                return None, r.status_code

            # Other statuses = retry
            time.sleep(5 + random.uniform(0.1, 0.5))
            attempts += 1

        except Exception as e:
            attempts += 1
            time.sleep(5 + random.uniform(0.1, 0.5))

    log_result(f'KB{kb_id:05d}', 'ERROR', 'Exceeded retry attempts')
    return None, "ERROR"


def is_public_kb_article(soup_obj):
    """
    Detects if the page contains a public-facing KB article.

    :param soup_obj: a BeautifulSoup object
    :return: True or False
    """

    # Check for structural page elements of a valid KB article
    if soup_obj.find(id="ctl00_ctl00_cpContent_cpContent_divBody"):
        return True
    else:
        return False


def log_result(kb_id, status, note=""):
    "Writes a row to the LOG_FILE wil the result of the request."

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{kb_id},{status},{note}\n")


def load_log_results():
    """
    Loads the LOG_FILE results and extracts the IDs of any articles that
    should be either Skipped or Retried

    :return: A 2-d tuple, where the first item is a Set of article IDs to
    retry, and the second item is a Set of article IDs to skip.
    """

    # Handle first time useage of the log
    if not LOG_FILE.exists():
        return set()

    # Declare variables to hold the logged items to skip or retry
    retry = set()
    skip = set()

    # Extract article IDs from the log to skip or retry
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            kb_id, status, *_ = line.strip().split(",")
            if status == "ERROR":
                retry.add(kb_id)
            else:
                skip.add(kb_id)

    return retry, skip


def scrape_articles(start, end):
    """
    Attempts to scrape all articles between the start and end IDs, and
    saves any public pages as HTML files.

    :param start: Integer representing the article ID to start scraping
    :param end: Integer representing the article ID to end scraping
    :return:
    """

    for kb_id in range(start, end + 1):

        formatted_id = f'KB{kb_id:05d}'
        html_path = HTML_DIR / f"{formatted_id}.html"

        # If the file already exists, skip
        if html_path.exists():
            continue

        # If scraping has already been attempted, skip
        retry, skip = load_log_results()
        if formatted_id in skip:
            continue

        response, status = get_article(kb_id)

        if status == 200 and response is not None:

            soup = BeautifulSoup(response.text, 'html.parser')

            if is_public_kb_article(soup):
                html_path.write_text(response.text, encoding="utf-8")
                log_result(formatted_id, 200, 'Public')
            else:
                log_result(formatted_id, 200, 'Private')

        time.sleep(5 + random.uniform(0.1, 0.5))

        if kb_id % 50 == 0:
            print(f"Completed scraping {kb_id} out of {end} pages.")


if __name__ == "__main__":
    scrape_articles(START_ON_ID,END_ON_ID)