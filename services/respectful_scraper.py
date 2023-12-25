import logging
import time
import requests
import json
from datetime import datetime
import os
from config.logging_config import setup_global_logger
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
from typing import Set
from dotenv import load_dotenv
from utils.utils import write_json_file

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
        # logging.info(f"Scraping {'IS' if is_allowed else 'IS NOT'} allowed for: {url}") # Turned off for now
        return is_allowed


def get_internal_links(base_url: str, soup: BeautifulSoup) -> Set[str]:
    """
    Extracts and returns all internal links on a webpage.

    This function finds all 'a' tags with href attributes in the provided BeautifulSoup object
    and checks if the links are internal, i.e., belonging to the same domain as the base URL.
    It normalizes both the base URL and each extracted link to ensure that differently formatted
    URLs pointing to the same resource are treated equally. The normalization includes
    lowercasing the scheme and netloc components of the URLs and resolving relative URLs. 
    The set data structure is used to avoid duplicate links.

    Parameters:
    - base_url (str): The base URL of the webpage. This is used to compare the domain of 
      the extracted links to identify internal links.
    - soup (BeautifulSoup): The parsed HTML content of the webpage.

    Returns:
    - set: A set of URLs (as strings) of internal links on the page. Each URL is normalized 
      to ensure uniqueness and correct identification of internal links.

    Notes:
    - This function is useful for web scraping tasks where understanding the internal link 
      structure of a website is necessary.
    - Normalization helps in treating different formats of the same URL as equivalent, thereby
      improving the accuracy of internal link identification.
    """
    links = set()
    base_parsed_url = urlparse(base_url)
    base_netloc = base_parsed_url.netloc.lower()

    # Normalize the base URL
    normalized_base_url = urlunparse(base_parsed_url._replace(scheme=base_parsed_url.scheme.lower(), netloc=base_netloc))

    for link in soup.find_all('a', href=True):
        href = link['href']
        joined_url = urljoin(base_url, href)
        parsed_joined_url = urlparse(joined_url)

        # Normalize the joined URL
        normalized_url = urlunparse(parsed_joined_url._replace(scheme=parsed_joined_url.scheme.lower(), netloc=parsed_joined_url.netloc.lower()))

        # Check if the link is internal using the normalized URLs
        if urlparse(normalized_url).netloc == urlparse(normalized_base_url).netloc:
            links.add(normalized_url)

    return links

def extract_text(soup: BeautifulSoup) -> str:
    """
    Efficiently extracts and returns the main textual content from a BeautifulSoup object.

    This function removes script, style, and other non-relevant elements, and 
    then extracts the remaining text using BeautifulSoup's get_text method, 
    which is more efficient than manual iteration.

    Parameters:
    soup (BeautifulSoup): A BeautifulSoup object parsed from a webpage.

    Returns:
    str: The extracted textual content as a single string.
    TODO: Add support for extracting text from other elements such as tables, lists, etc.
    TODO: Handle dynamic content loaded via JavaScript.
    """
    # Remove non-relevant elements
    for element in soup(['script', 'style', 'head', 'title', 'meta', '[document]']):
        element.decompose()

    text_content = soup.get_text(separator=' ', strip=True)

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

    Raises:
    - This function does not raise any exceptions by itself but depends on the correct creation of the BeautifulSoup object.

    Notes:
    - This function is designed to be used in conjunction with a web scraping routine where the HTML content
      has already been fetched and parsed.
    - The quality and structure of the extracted data highly depend on the structure of the HTML content.
    """
    text = extract_text(soup)
    metadata = extract_metadata(soup)

    combined_data = {
        'url': url,
        'text': text,
        'scrape_timestamp': datetime.now().isoformat(),
        **metadata,
    }

    return combined_data


def scrape_website(start_urls, user_agent: str, max_depth: int = 3, output_dir: str = 'data/scraped_data', request_delay: float = 1) -> dict:
    """
    Scrapes websites starting from a list of given URLs or a single URL, respecting robots.txt rules, and retrieves text and metadata from each page.
    Note: The usage of a set for managing URLs to visit does not maintain the order of URLs, thereby not supporting ordered scraping methods like breadth-first or depth-first search.

    Parameters:
    - start_urls (str or list): The initial URL(s) to start scraping from. Can be a single URL or a list of URLs.
    - user_agent (str): The user agent string to be used for HTTP requests and robots.txt compliance.
    - max_depth (int, optional): The maximum depth to follow internal links for scraping. Defaults to 3.
    - output_dir (str, optional): The directory where the scraped data JSON files will be stored. Defaults to 'data/scraped_data'.
    - request_delay (float, optional): Minimal delay in seconds between requests. Defaults to 1 second.

    Returns:
    - dict: A summary of the scraping process, including counts of pages scraped and visited, and the path to the output file.

    Raises:
    - requests.RequestException: If an error occurs during the HTTP request to fetch webpage content.
    - Exception: For any other unexpected errors during the scraping process.
    TODO: Fetch and parse sitemap.xml to get a list of URLs to scrape.
    TODO: Fetch and respect crawl-delay from robots.txt.
    TODO: Consider refactoring to make the function more modular and reusable.
    TODO: Add support for scraping dynamic content loaded via JavaScript.
    """

    if not isinstance(start_urls, list):
        start_urls = [start_urls]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract domain from the first URL in the list
    first_domain = urlparse(start_urls[0]).netloc.replace('.', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'scraped_{first_domain}_{timestamp}.json')

    robots_cache = {}
    visited = set()
    to_visit = {(url, 0) for url in start_urls}
    scraped_data = []

    with requests.Session() as session:
        session.headers = {'User-Agent': user_agent}

        while to_visit:
            url, depth = to_visit.pop()
            if depth > max_depth or url in visited:
                continue

            scraping_allowed = can_fetch(url, user_agent=user_agent, robots_cache=robots_cache)

            if scraping_allowed:
                try:
                    logging.info(f"Scraping {url} - depth: {depth}, visited: {len(visited)}, to_visit: {len(to_visit)}")
                    response = session.get(url)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        page_data = scrape_page(soup, url)
                        scraped_data.append(page_data)
                        visited.add(url)

                        internal_links = get_internal_links(url, soup)
                        for link in internal_links:
                            if link not in visited:
                                to_visit.add((link, depth + 1))
                        time.sleep(request_delay)
                    else:
                        logging.error(f"Failed to fetch {url}: HTTP status code {response.status_code}")
                except Exception as e:
                    logging.error(f"Error scraping {url}: {e}")
            else:
                visited.add(url)

    write_json_file(scraped_data, output_file)
    
    return {
        'total_scraped': len(scraped_data),
        'total_visited': len(visited),
        'output_file': output_file
    }



def main():
    logging.basicConfig(level=logging.INFO)
    setup_global_logger() 
    # Load the .env file
    load_dotenv()
    
    # Get variables from .env file
    user_agent = os.getenv('USER_AGENT')
    start_url = os.getenv('START_URL')

    max_depth = 10  # Define the maximum depth for scraping
    output_dir = 'data/scraped_data'  # Directory where the scraped data will be stored
    request_delay = 0.3  # Minimal delay in seconds between requests

    # Start the scraping process
    scraping_summary = scrape_website(start_url, user_agent, max_depth, output_dir, request_delay)

    # Print the summary of the scraping process
    print(f"Scraping completed. Total pages scraped: {scraping_summary['total_scraped']}")
    print(f"Total unique pages visited: {scraping_summary['total_visited']}")
    print(f"Scraped data stored in: {scraping_summary['output_file']}")

if __name__ == "__main__":
    main()