# TODO: Organize imports according to PEP 8 standards: standard library imports, 
# followed by related third-party imports, and then local application/library specific imports.

import logging
import time
import requests
import json
from datetime import datetime
import os
import copy
from config.logging_config import setup_global_logger
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
from typing import Set
from dotenv import load_dotenv
from utils.utils import write_json_file, read_yaml_file


# TODO: Make sure at runtime that the requests are only HTTPS


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

def extract_text(soup: BeautifulSoup, content_selectors=None) -> str:
    """
    Extracts and returns the main textual content from a BeautifulSoup object, focusing on specified main content areas.
    It avoids headers, footers, menus, and other non-essential sections. If the main content is not found using the provided
    selectors, the function returns an empty string.

    Parameters:
    soup (BeautifulSoup): A BeautifulSoup object parsed from a webpage.
    content_selectors (list, optional): A list of CSS selectors for targeting main content. If not provided, default selectors are used.

    Returns:
    str: The extracted textual content as a single string, or an empty string if main content is not found.
    TODO: Explictily handle dynamic content loaded via JavaScript.

    """
    # Default selectors for main content if none are provided
    if content_selectors is None:
        content_selectors = [
        'article',
        '.article-content',
        '.article_body',
        '.post-content',
        '.post-body',
        '.main-content',
        '.post',
        'section.content',
        'main[role="main"]',
        '.info',
        'aside',
        '#main',
        'div.content',
        '.text-content',
        '.entry-content',
    ]  

    main_content = []
    for selector in content_selectors:
        elements = soup.select(selector)
        for element in elements:
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
    metadata = extract_metadata(soup)

    combined_data = {
        'url': url,
        'text': text,
        'scrape_timestamp': datetime.now().isoformat(),
        **metadata,
    }

    return combined_data


def scrape_website(start_urls, user_agent: str, max_depth: int = 3, request_delay: float = 1) -> dict:
    """
    Scrapes websites starting from a list of given URLs or a single URL, respecting robots.txt rules, and retrieves text and metadata from each page.
    Note: The usage of a set for managing URLs to visit does not maintain the order of URLs, thereby not supporting ordered scraping methods like breadth-first or depth-first search.

    Parameters:
    - start_urls (str or list): The initial URL(s) to start scraping from. Can be a single URL or a list of URLs.
    - user_agent (str): The user agent string to be used for HTTP requests and robots.txt compliance.
    - max_depth (int, optional): The maximum depth to follow internal links for scraping. Defaults to 3.
    - request_delay (float, optional): Minimal delay in seconds between requests. Defaults to 1 second.

    Returns:
    - dict: A summary of the scraping process, including counts of pages scraped and visited.

    Raises:
    - requests.RequestException: If an error occurs during the HTTP request to fetch webpage content.
    - Exception: For any other unexpected errors during the scraping process.
    TODO: Fetch and parse sitemap.xml to get a list of URLs to scrape.
    TODO: Fetch and respect crawl-delay from robots.txt.
    TODO: Consider refactoring to make the function more modular and reusable.
    TODO: Add support for scraping dynamic content loaded via JavaScript.
    TODO: Concatenate meta description with the fetched text
    """

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

    scraping_result = {
        'scraped_data': scraped_data,
        'total_scraped': len(scraped_data),
        'total_visited': len(visited),
    }

    logging.info(f"Scraping completed. Total pages scraped: {scraping_result['total_scraped']}")
    logging.info(f"Total unique pages visited: {scraping_result['total_visited']}")

    return scraping_result

# TODO: rozważyć format tradycyjny tego docstringa typu args, return itp
# TODO: zadbać o logging poprawnie wykonanego skryptu, progres

def main():
    """
    Main function to initiate the website scraping process.
    It only scrapes sites within the same domain as the starting URL.

    The function sets up logging, reads configuration parameters from a YAML file, 
    and then initiates the scraping process based on these parameters. 
    It concludes by logging the summary of the scraping results.

    Configuration Parameters:
    - `scraping_start_url` (str): The URL where the scraping process begins.
    - `scraping_user_agent` (str): The User-Agent string to be used for HTTP requests. That is, how you introduce youself to the server.
    - `scraping_max_depth` (int): This parameter determines how deep the scraper will navigate from the 
      starting URL, following links within the site. It's an integer representing the levels of depth the 
      scraper will traverse.

      Explanation:
      - Depth Level 0: The scraper starts at the `scraping_start_url`. No links from this page are followed.
      - Depth Level 1: The scraper follows links found on the `scraping_start_url` page, scraping the 
        content of these linked pages.
      - Depth Level 2: In addition to level 1, the scraper also reaches links found on the "Depth Level 1" pages,
        scraping their content as well, and so on.

      For example, if `scraping_max_depth` is set to 2, the scraper will scrape the starting page, 
      the pages linked directly from the starting page, and the pages linked from those pages.
    - `scraping_request_delay` (float): Delay (in seconds) between consecutive HTTP requests to avoid overloading the server.
    - `scraping_output_dir` (str): Directory path where the scraped data will be stored.

    Logs:
    After completion, the function logs:
    - Total number of pages scraped.
    - Total number of unique pages visited.
    - The path to the file where the scraped data is stored.
    """
    # TODO: setup logging level as a parameter in the config file
    logging.basicConfig(level=logging.INFO)
    setup_global_logger() 

    # Configuration parameters
    config = read_yaml_file('config/parameters.yml')
    website_scraper_config = config['website_scraper']

    start_url = website_scraper_config.get('scraping_start_url')
    user_agent = website_scraper_config.get('scraping_user_agent')
    max_depth = website_scraper_config.get('scraping_max_depth')
    request_delay = website_scraper_config.get('scraping_request_delay')
    output_dir = website_scraper_config.get('scraping_output_dir')

    # Scraping
    scraping_result = scrape_website(start_url, user_agent, max_depth, output_dir, request_delay)
    write_json_file(data=scraping_result['scraped_data'], output_file_path=scraping_result['output_file'])


if __name__ == "__main__":
    main()