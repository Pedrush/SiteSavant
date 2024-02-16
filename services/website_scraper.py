"""
This script is designed for scraping websites in a respectful manner, adhering to robots.txt policies
and extracting  information from web pages. It leverages the 'BeautifulSoup' for parsing HTML content and 'requests' for HTTP interactions.

Functional Overview:

- Sets up a global logging configuration to monitor and log the scraping process.
- Fetches and parses robots.txt files to respect website crawling restrictions.
- Determines scraping permissions for URLs based on the robots.txt file, ensuring compliance with website policies.
- Extracts internal links, text content, and metadata from web pages, focusing on semantic HTML elements to capture relevant information.
- Utilizes a session-based approach with requests to efficiently manage HTTP connections and headers, reducing overhead and improving speed.
- Supports configurable depth-based scraping, allowing users to specify how deep the scraper should navigate from the starting URLs.
- Implements delay between requests to respect server load and prevent unethical, aggressive scraping behavior.
- Saves scraped data in a structured JSON format.

Components:
- fetch_robots_txt: Fetches and parses a website's robots.txt file, determining access permissions for the scraper.
- can_fetch: Checks whether scraping a particular URL is allowed, using the parsed robots.txt rules.
- get_internal_links: Extracts all internal links from a webpage, enabling depth-based scraping.
- extract_text: Captures the main textual content from a webpage, excluding common non-content areas like headers and footers.
- extract_metadata: Extracts basic metadata such as the page title, meta description, and headings from a webpage.
- scrape_page: Combines text and metadata extraction to gather comprehensive information from a single webpage.
- scrape_website: Orchestrates the scraping process, managing URL queues, respecting robots.txt rules, and collecting data from multiple pages.
- Utility functions for reading configuration, writing JSON files, and generating timestamps support the scraping workflow.

Usage:
Can be used as a standalone module. Additionally, the functions are designed to integrate with a larger processing pipeline, 
as demonstrated in the main orchestrator script (main.py) within this project."""

# Standard library imports
import copy
import logging
import time
from datetime import datetime
from typing import Set
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

# Related third-party imports
from bs4 import BeautifulSoup
import requests

# Local application/library specific imports
from config.logging_config import setup_global_logger
from utils.utils import write_json_file, generate_timestamp, read_yaml_file


def fetch_robots_txt(robots_url: str, user_agent: str) -> RobotFileParser:
    """
    Fetch and parse the robots.txt file for a given URL.

    Parameters:
    robots_url (str): The URL of the robots.txt file.
    user_agent (str): The user agent string to use for the request.

    Returns:
    RobotFileParser: A RobotFileParser instance with parsed data or None in case of failure.
    """
    try:
        headers = {'User-Agent': user_agent}
        response = requests.get(robots_url, headers=headers)
        rp = RobotFileParser()
        if response.status_code in [401, 403]:
            logging.warning(f"Access denied to {robots_url}")
            rp.disallow_all = True
            return rp
        elif 400 <= response.status_code < 500:
            logging.warning(f"No restrictions for {robots_url}. Response code: {response.status_code}")
            rp.allow_all = True
            return rp
        else:
            rp.parse(response.text.splitlines())
            return rp
    except requests.RequestException as e:
        logging.error(f"Error fetching {robots_url}: {e}")
        return None

def can_fetch(url: str, user_agent: str, robots_cache: dict = None) -> bool:
    """
    Determines if scraping a given URL is allowed based on the robots.txt file of the host.

    Parameters:
    url (str): The URL to check for scraping permissions.
    user_agent (str): The user agent to check for in the robots.txt file.
    robots_cache (dict, optional): Cache for storing robots.txt data.

    Returns:
    bool: True if scraping is allowed, False if it is not allowed or if an error occurs
          in fetching or parsing the robots.txt file.

    Note:
    In case of an error (such as failure in fetching or parsing robots.txt), the function logs the error
    and returns False. This is to ensure the continuity of the scraping process without interruption
    while still respecting the potential restrictions of the target website.
    TODO: "http://example.com/privatepage.html" will wrongly be True when "User-agent: *\nDisallow: /private*/"
    """
    if robots_cache is None:
        robots_cache = {}

    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        logging.error(f"Invalid URL: {url}")
        return False

    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

    if robots_url not in robots_cache:
        robots_cache[robots_url] = fetch_robots_txt(robots_url, user_agent)

    rp = robots_cache[robots_url]
    if rp is None:
        logging.error(f"An UNEXPECTED error occurred during fetching or parsing robots.txt for {robots_url}")
        return False
    else:
        is_allowed = rp.can_fetch(user_agent, url)
        return is_allowed


def get_internal_links(base_url: str, soup: BeautifulSoup) -> Set[str]:
    """
    Extracts and returns all internal links on a webpage.

    This function finds all 'a' tags with href attributes in the provided BeautifulSoup object
    and checks if the links are internal, i.e., belonging to the same domain as the base URL.
    It normalizes both the base URL and each extracted link to ensure that differently formatted
    URLs pointing to the same resource are treated equally. The normalization includes
    lowercasing the scheme and netloc components of the URLs, resolving relative URLs, and 
    removing fragments. The set data structure is used to avoid duplicate links.

    Parameters:
    - base_url (str): The base URL of the webpage. This is used to compare the domain of 
      the extracted links to identify internal links.
    - soup (BeautifulSoup): The parsed HTML content of the webpage.

    Returns:
    - set: A set of URLs (as strings) of internal links on the page. Each URL is normalized 
      to ensure uniqueness and correct identification of internal links.
    """
    links = set()
    base_parsed_url = urlparse(base_url)
    base_netloc = base_parsed_url.netloc.lower()

    # Normalize the base URL, including removing the fragment
    normalized_base_url = urlunparse(base_parsed_url._replace(scheme=base_parsed_url.scheme.lower(), netloc=base_netloc, fragment=''))

    for link in soup.find_all('a', href=True):
        href = link['href']
        joined_url = urljoin(base_url, href)
        parsed_joined_url = urlparse(joined_url)

        # Normalize the joined URL, including removing the fragment
        normalized_url = urlunparse(parsed_joined_url._replace(scheme=parsed_joined_url.scheme.lower(), netloc=parsed_joined_url.netloc.lower(), fragment=''))

        # Check if the link is internal using the normalized URLs
        if urlparse(normalized_url).netloc == urlparse(normalized_base_url).netloc:
            links.add(normalized_url)

    return links

def extract_text(soup: BeautifulSoup) -> str:
    """
    Extracts and returns the main textual content from a BeautifulSoup object, focusing on semantic HTML elements.
    It avoids headers, footers, menus, and other non-essential sections by not selecting these elements for text extraction.

    Parameters:
    soup (BeautifulSoup): A BeautifulSoup object parsed from a webpage.

    Returns:
    str: The extracted textual content as a single string, or an empty string if main content is not found.
    """
    # List of semantic HTML elements typically used for main content
    content_elements = ['article', 'main', 'section']

    # Elements to exclude from the text extraction, commonly non-content areas
    exclude_elements = ['header', 'footer', 'nav', 'aside', 'script', 'style']

    main_content = []
    for tag in content_elements:
        for element in soup.find_all(tag):
            for exclude_tag in exclude_elements:
                for excluded in element.find_all(exclude_tag):
                    excluded.decompose()
            # Extract and clean text from the remaining content
            main_content.append(element.get_text(separator=' ', strip=True))

    if not main_content:
        return ""

    # Combine all text from the main content areas
    text_content = '\n\n'.join(main_content)

    return text_content


def extract_metadata(soup: BeautifulSoup) -> dict:
    """
    Extracts and returns metadata from a BeautifulSoup object.

    This function captures the title, meta description, and headings (h1 to h6) from the webpage.
    It handles cases where the meta tag might not have a 'content' attribute or the title tag is absent.

    Parameters:
    soup (BeautifulSoup): A BeautifulSoup object parsed from a webpage.

    Returns:
    dict: A dictionary containing the extracted metadata, including the title, meta description, and headings.
    TODO: Add support for other metadata such as keywords, author, etc.
    """
    # Extract title
    title = soup.title.string if soup.title else None

    # Extract meta description
    meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
    meta_description = meta_desc_tag['content'] if meta_desc_tag and 'content' in meta_desc_tag.attrs else None

    # Extract headings
    headings = [heading.get_text().strip() for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]

    metadata = {
        'title': title,
        'meta_description': meta_description,
        'headings': headings
    }
    return metadata


def scrape_page(soup: BeautifulSoup, url: str) -> dict:
    """
    Extracts textual content and metadata from a BeautifulSoup object representing a webpage.

    This function processes the HTML content encapsulated by the BeautifulSoup object to extract
    text and metadata such as titles, headers, and other relevant information. It's designed to work
    with webpages where the relevant data is embedded in the HTML structure.

    Parameters:
    - soup (BeautifulSoup): A BeautifulSoup object containing the parsed HTML content of the page.
    - url (str): The URL of the webpage, used for reference in the returned data.

    Returns:
    - dict: A dictionary with keys 'url', 'text', 'scrape_timestamp', and other metadata extracted from the page.
      'text' contains the main textual content, and 'scrape_timestamp' is the datetime when the scrape occurred.
      Other metadata are extracted based on the structure of the webpage.

    TODO: make sure the unicode characters are decoded before writing to JSON or before indexing in Pinecone.
    """

    metadata = extract_metadata(soup)

    exclusion_selectors = [
        '.ad', '.advert', '.advertisement', '#ads', '.ads-banner',
        'header', '.header', '#header',
        '.breadcrumbs', '#breadcrumbs',
        '.comments', '#comments', '.comment-section',
        '.social-links', '.social-media', '#social-media',
        'footer', '.footer', '.legal', '.disclaimer', '.terms', '.privacy-policy',
        '#modal', '.modal', '.popup', '#popup',
        'nav', '.navigation', 'nav.pagination', '.page-navigation',
        '.author-info', '#author-bio',
        '.sidebar', '.menu'
    ]

    for selector in exclusion_selectors:
        for elem in soup.select(selector):
            elem.extract()

    text = extract_text(soup)

    combined_data = {
        'url': url,
        'text': text,
        'scrape_timestamp': datetime.now().isoformat(),
        **metadata,
    }

    return combined_data


def scrape_website(start_urls, user_agent: str, max_depth: int = 2, request_delay: float = 1) -> dict:
    """
    Scrapes websites starting from a list of given URLs or a single URL, respecting robots.txt rules, and retrieves text and metadata from each page.
    Note: The usage of a set for managing URLs to visit does not maintain the order of URLs, thereby not supporting ordered scraping methods like breadth-first or depth-first search.

    Parameters:
    - start_urls (str or list): The initial URL(s) to start scraping from. Can be a single URL or a list of URLs.
    - user_agent (str): The user agent string to be used for HTTP requests and robots.txt compliance.
    - request_delay (float, optional): Minimal delay in seconds between requests. Defaults to 1 second.
    - max_depth (int, optional): The maximum depth to follow internal links for scraping. Defaults to 2. Examples:
      - Depth Level 0: The scraper starts at the `scraping_start_url`. No links from this page are followed.
      - Depth Level 1: The scraper follows links found on the `scraping_start_url` page, scraping the 
        content of these linked pages.
      - Depth Level 2: In addition to level 1, the scraper also reaches links found on the "Depth Level 1" pages,
        scraping their content as well, and so on.

    Returns:
    - list: A list of dictionaries, each containing the scraped data from a single page. Each dictionary contains the URL, text, and metadata of the page.

    Raises:
    - requests.RequestException: If an error occurs during the HTTP request to fetch webpage content.
    - Exception: For any other unexpected errors during the scraping process.
    TODO: Make sure the consistent type of start_urls parameter is enforced.
    TODO: Fetch and parse sitemap.xml to get a list of URLs to scrape.
    TODO: Fetch and respect crawl-delay from robots.txt.
    TODO: Consider refactoring to make the function more modular and reusable.
    TODO: Add support for scraping dynamic content loaded via JavaScript.
    TODO: Make sure at runtime that the requests are only HTTPS.
    """

    if not start_urls.startswith(('http://', 'https://')):
        start_urls = 'https://' + start_urls

    if not isinstance(start_urls, list):
        start_urls = [start_urls]

    robots_cache = {}
    visited = set()
    to_visit = {(url, 0) for url in start_urls}
    scraped_data = []

    with requests.Session() as session:
        session.headers = {'User-Agent': user_agent}

        while to_visit:
            url, depth = to_visit.pop()
            if url in visited:
                continue

            scraping_allowed = can_fetch(url, user_agent=user_agent, robots_cache=robots_cache)

            if scraping_allowed:
                try:
                    logging.info(f"Scraping {url} - depth: {depth}, visited: {len(visited)}, to_visit: {len(to_visit)}")
                    response = session.get(url)
                    if response.status_code == 200:
                        original_soup = BeautifulSoup(response.content, 'html.parser')
                        soup = copy.deepcopy(original_soup)
                        page_data = scrape_page(soup, url)
                        scraped_data.append(page_data)
                        visited.add(url)

                        if depth < max_depth:
                            internal_links = get_internal_links(url, original_soup)
                            for link in internal_links:
                                if link not in visited:
                                    to_visit.add((link, depth + 1))
                        time.sleep(request_delay)
                    else:
                        logging.error(f"Failed to fetch {url}: HTTP status code {response.status_code}")
                except Exception as e:
                    logging.error(f"Error scraping {url}: {e}")
            else:
                logging.warning(f"Scraping not allowed for {url}")
                visited.add(url)

    logging.info(f"Scraping completed. Total pages scraped: {len(scraped_data)}")
    logging.info(f"Total unique pages visited: {len(visited)}")

    return scraped_data

def main():
    """
    Demonstrates the website scraping process using predefined configuration parameters.
    
    Steps:    
    1. Sets up logging for monitoring the scraping progress.
    2. Reads configuration from 'config/parameters.yml', which includes the start URL, user agent, 
      max depth for scraping, request delay, and output directory for storing scraped data.
    3. Initiates the scraping process and stores the results in the specified output directory.
    4. Logs the total number of pages scraped and unique pages visited.
    5. Saves the scraped data to a JSON file in the specified directory.
    """
    # Config
    logging.basicConfig(level=logging.INFO)
    setup_global_logger()
    timestamp = generate_timestamp()

    all_parameters = read_yaml_file('config/parameters.yml')
    config = all_parameters['main_config']
    file_paths = all_parameters['file_paths']

    # Scraping website
    scraped_data = scrape_website(**config['website_scraper'])
    write_json_file(
        data=scraped_data,
        file_path=file_paths['website_scraper']['output_scraping_file_path'],
        timestamp=timestamp
        )


if __name__ == "__main__":
    main()