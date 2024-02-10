import pytest
import requests
import requests_mock
from urllib.robotparser import RobotFileParser
from services.website_scraper import fetch_robots_txt, can_fetch, get_internal_links, extract_text, extract_metadata, scrape_page, scrape_website
from bs4 import BeautifulSoup
from unittest.mock import Mock
from datetime import datetime
from unittest.mock import patch

# Test for a successful fetch with actual rules
def test_fetch_robots_txt_success():
    robots_url = "https://example.com/robots.txt"
    user_agent = "TestAgent"
    with requests_mock.Mocker() as m:
        m.get(robots_url, text="User-agent: *\nDisallow: /private")
        result = fetch_robots_txt(robots_url, user_agent)
        assert isinstance(result, RobotFileParser)
        result.parse(robots_url)
        assert not result.can_fetch(user_agent, "https://example.com/private")

# Test for access denied (403, 401 HTTP status codes)
@pytest.mark.parametrize("status_code", [401, 403])
def test_fetch_robots_txt_access_denied(status_code):
    robots_url = "https://example.com/robots.txt"
    user_agent = "TestAgent"
    with requests_mock.Mocker() as m:
        m.get(robots_url, status_code=status_code)
        result = fetch_robots_txt(robots_url, user_agent)
        assert isinstance(result, RobotFileParser)
        assert result.disallow_all

# Test for other client errors (4XX except 401, 403)
def test_fetch_robots_txt_client_error():
    robots_url = "https://example.com/robots.txt"
    user_agent = "TestAgent"
    with requests_mock.Mocker() as m:
        m.get(robots_url, status_code=404)
        result = fetch_robots_txt(robots_url, user_agent)
        assert isinstance(result, RobotFileParser)
        assert result.allow_all

# Test for network/request exceptions
def test_fetch_robots_txt_exception():
    robots_url = "https://badurl.com/robots.txt"
    user_agent = "TestAgent"
    with requests_mock.Mocker() as m:
        m.get(robots_url, exc=requests.RequestException)
        result = fetch_robots_txt(robots_url, user_agent)
        assert result is None

@pytest.fixture
def mock_robot_parser_allow():
    rp = RobotFileParser()
    rp.allow_all = True
    return rp

@pytest.fixture
def mock_robot_parser_disallow():
    rp = RobotFileParser()
    rp.disallow_all = True
    return rp

# Test for allowed URL
def test_can_fetch_allowed(mocker, mock_robot_parser_allow):
    mocker.patch('services.website_scraper.fetch_robots_txt', return_value=mock_robot_parser_allow)
    assert can_fetch("http://example.com/publicpage.html", "TestAgent") is True

# Test for disallowed URL
def test_can_fetch_disallowed(mocker, mock_robot_parser_disallow):
    mocker.patch('services.website_scraper.fetch_robots_txt', return_value=mock_robot_parser_disallow)
    assert can_fetch("http://example.com/privatepage.html", "TestAgent") is False

# Test for invalid URL
def test_can_fetch_invalid_url():
    assert can_fetch("ftp://example.com", "TestAgent") is False

# Test using cache
def test_can_fetch_with_cache(mocker):
    mock_rp = mocker.Mock(spec=RobotFileParser)
    mock_rp.can_fetch.return_value = True
    cache = {"http://example.com/robots.txt": mock_rp}
    assert can_fetch("http://example.com/publicpage.html", "TestAgent", robots_cache=cache) is True
    mock_rp.can_fetch.assert_called_once_with("TestAgent", "http://example.com/publicpage.html")

# Test for error fetching robots.txt
def test_can_fetch_error_fetching(mocker):
    mocker.patch('services.website_scraper.fetch_robots_txt', return_value=None)
    assert can_fetch("http://example.com/shouldfail.html", "TestAgent") is False

# Utility function to create a BeautifulSoup object
def get_soup(html_content):
    return BeautifulSoup(html_content, 'html.parser')

# Test case for extracting internal links
def test_get_internal_links():
    base_url = "http://example.com"
    soup = get_soup(HTML_CONTENT)
    internal_links = get_internal_links(base_url, soup)
    
    expected_links = set([
        "http://example.com",
        "http://example.com/internal/link1",
        "http://example.com/internal/link2",
        "https://example.com/internal/link3",
        "http://example.com/internal/link4",  # Assuming the base_url scheme is http
        "https://example.com/internal/link5",  # Excluding fragment
        "https://example.com/internal/link6?query=param",  # Including query
        "http://example.com/internal/link7/",
    ])
    
    assert internal_links == expected_links

# Mock HTML content for testing
HTML_CONTENT = """
<html>
<head><title>Test Page</title></head>
<body>
    <a href="/internal/link1">Internal Link 1</a>
    <a href="http://example.com/internal/link2">Internal Link 2</a>
    <a href="https://example.com/internal/link3">Internal Link 3</a>
    <a href="//example.com/internal/link4">Protocol Relative URL</a>
    <a href="http://external.com/external/link1">External Link 1</a>
    <a href="https://example.com/internal/link5#fragment">Internal Link with Fragment</a>
    <a href="https://example.com/internal/link6?query=param">Internal Link with Query</a>
    <a href="/internal/link7/">Internal Link with Trailing Slash</a>
    <a href="javascript:void(0);">JavaScript Link</a>
    <a href="#">Empty Link</a>
</body>
</html>
"""


def test_basic_extraction():
    html_content = "<main>Some main content here.</main>"
    soup = BeautifulSoup(html_content, 'html.parser')
    assert extract_text(soup) == "Some main content here."

def test_complex_structure():
    html_content = """
    <article>Article content here.</article>
    <main>Main content here.</main>
    <section>Section content here.</section>
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    expected = "Article content here.\n\nMain content here.\n\nSection content here."
    assert extract_text(soup) == expected

def test_exclusion_of_non_content_areas():
    html_content = """
    <main>Main content here. <nav>Navigation</nav> <footer>Footer content</footer></main>
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    assert extract_text(soup) == "Main content here."

# TODO: Fix the test
def test_nested_content_elements():
    html_content = """
    <main>Main content here <section>Nested section content</section> and more main content.</main>
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    assert extract_text(soup) == "Main content here Nested section content and more main content."

def test_empty_content():
    html_content = "<div>Only non-semantic content</div>"
    soup = BeautifulSoup(html_content, 'html.parser')
    assert extract_text(soup) == ""

# TODO: Fix the test
def test_separator_and_stripping():
    html_content = """
    <main>
        Main content here
        <section>   Nested section content </section>
        and more main content.
    </main>
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    expected = "Main content here Nested section content and more main content."
    assert extract_text(soup) == expected

def test_full_metadata_present():
    html_content = """
    <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="This is a test page for metadata extraction.">
        </head>
        <body>
            <h1>Main Heading</h1>
            <h2>Subheading</h2>
        </body>
    </html>
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    metadata = extract_metadata(soup)
    assert metadata == {
        'title': 'Test Page',
        'meta_description': 'This is a test page for metadata extraction.',
        'headings': ['Main Heading', 'Subheading']
    }

def test_missing_title():
    html_content = """
    <html>
        <head>
            <meta name="description" content="No title here.">
        </head>
    </html>
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    metadata = extract_metadata(soup)
    assert metadata['title'] is None

def test_missing_meta_description():
    html_content = """
    <html>
        <head>
            <title>Missing Description</title>
        </head>
    </html>
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    metadata = extract_metadata(soup)
    assert metadata['meta_description'] is None

def test_headings_only():
    html_content = """
    <html>
        <body>
            <h1>Heading One</h1>
            <h2>Heading Two</h2>
        </body>
    </html>
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    metadata = extract_metadata(soup)
    assert metadata['title'] is None
    assert metadata['meta_description'] is None
    assert metadata['headings'] == ['Heading One', 'Heading Two']

def test_no_metadata():
    html_content = "<html><body>Just a test.</body></html>"
    soup = BeautifulSoup(html_content, 'html.parser')
    metadata = extract_metadata(soup)
    assert metadata == {'title': None, 'meta_description': None, 'headings': []}

def test_invalid_meta_description():
    html_content = """
    <html>
        <head>
            <meta name="description">
        </head>
    </html>
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    metadata = extract_metadata(soup)
    assert metadata['meta_description'] is None

def test_mixed_headings():
    html_content = """
    <html>
        <body>
            <h3>Third Level</h3>
            <h1>First Level</h1>
            <h2>Second Level</h2>
        </body>
    </html>
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    metadata = extract_metadata(soup)
    assert metadata['headings'] == ['Third Level', 'First Level', 'Second Level']

# TODO: Fix the test
def test_basic_page_scraping():
    html_content = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Header</h1>
            <p>This is a test paragraph.</p>
        </body>
    </html>
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    result = scrape_page(soup, "http://example.com")
    assert 'This is a test paragraph.' in result['text']
    assert result['url'] == "http://example.com"
    assert 'Test Page' == result['title']

# TODO: Fix the test
def test_exclusion_of_elements():
    html_content = """
    <html>
        <body>
            <div class="advertisement">This is an ad.</div>
            <p>This paragraph should remain.</p>
        </body>
    </html>
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    result = scrape_page(soup, "http://example.com")
    assert 'This is an ad.' not in result['text']
    assert 'This paragraph should remain.' in result['text']

def test_url_and_timestamp_inclusion():
    url = "http://example.com"
    soup = BeautifulSoup("<html></html>", 'html.parser')
    result = scrape_page(soup, url)
    assert result['url'] == url
    # Assuming the test runs fast enough for the timestamp to be within the same minute
    assert datetime.now().isoformat()[:16] == result['scrape_timestamp'][:16]

def test_metadata_extraction():
    html_content = """
    <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="Description for testing.">
        </head>
        <body>
            <h1>Main Heading</h1>
        </body>
    </html>
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    result = scrape_page(soup, "http://example.com")
    assert result['title'] == "Test Page"
    assert result['meta_description'] == "Description for testing."
    assert "Main Heading" in result['headings']

def test_handling_of_missing_elements():
    html_content = "<html><body><p>No metadata here.</p></body></html>"
    soup = BeautifulSoup(html_content, 'html.parser')
    result = scrape_page(soup, "http://example.com")
    assert result['title'] is None
    assert result['meta_description'] is None
    assert result['headings'] == []


@pytest.fixture
def mock_environment():
    with requests_mock.Mocker() as m:
        yield m

def test_single_start_url(mock_environment):
    mock_url = "https://example.com"
    mock_environment.get(mock_url, text="<html></html>", status_code=200)
    with patch('services.website_scraper.can_fetch', return_value=True), \
         patch('services.website_scraper.scrape_page', return_value={'url': mock_url, 'text': '', 'metadata': {}}), \
         patch('services.website_scraper.get_internal_links', return_value=set()):
        result = scrape_website(mock_url, "TestAgent")
        assert len(result) == 1
        assert result[0]['url'] == mock_url

def test_respect_robots_txt(mock_environment):
    mock_url = "https://example.com"
    mock_environment.get(mock_url, text="<html></html>", status_code=200)
    with patch('services.website_scraper.can_fetch', return_value=False):
        result = scrape_website(mock_url, "TestAgent")
        assert len(result) == 0  # No data scraped due to robots.txt rules

def test_error_handling(mock_environment):
    mock_url = "https://example.com"
    mock_environment.get(mock_url, text="Error", status_code=500)
    with patch('services.website_scraper.can_fetch', return_value=True), \
         patch('services.website_scraper.scrape_page') as mock_scrape_page:
        mock_scrape_page.side_effect = Exception("Test exception")
        # Assuming scrape_website is properly catching exceptions,
        # this test may need to verify logging or the absence of a crash.
        result = scrape_website(mock_url, "TestAgent")
        # Depending on how scrape_website handles errors, you may want to assert something here.