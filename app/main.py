"""
SiteSavant - A chatbot for any website on-demand.

SiteSavant Main Application Orchestrator:
This script acts as the central orchestrator for the SiteSavant application, handling the entire lifecycle of chatbot interactions
based on website data. It integrates various services for web scraping, data processing, embeddings creation and deduplication,
indexing, and chatbot interactions. The application supports custom configurations and command-line arguments to tailor
the processing and interaction flow.

Key Components:
- Web scraping to collect data from specified URLs.
- Data processing including text embeddings creation, deduplication, and indexing for efficient retrieval.
- A chatbot that provides answers based on the processed website data.

CLI Commands:
- scrape: Initiates web scraping and data preparation. Requires a URL as an argument.
  Usage: `sitesavant scrape [URL]` - Scrapes the specified URL and prepares data for chatbot interaction.
- chat: Activates the chatbot using previously prepared data.
  Usage: `sitesavant chat` - Starts chatbot interaction without requiring additional data preparation.

Environment Configuration:
The application requires certain environment variables for accessing external APIs (COHERE_API_KEY, PINECONE_API_KEY, OPENAI_API_KEY). Ensure these are set before running the script.

Usage:
1. Prepare data by running the scrape command with the desired URL.
2. Interact with the chatbot using the chat command.

Example Commands:
- sitesavant scrape example.com
- sitesavant chat
"""

# Standard library imports
import argparse
import logging
import os
from typing import Any, Dict

# Related third-party imports
from dotenv import load_dotenv

# Local application/library specific imports
from config.logging_config import setup_global_logger
from services.website_scraper import scrape_website
from services.embeddings_creator import create_embeddings
from services.embeddings_deduplicator import deduplicate_embeddings, process_and_sort_duplicates
from services.embeddings_indexer import index_records
from services.query_handler import process_query, parse_query_results
from services.chatbot_interactor import generate_chat_response
from utils.utils import write_json_file, save_embeddings_and_metadata, prepend_title_and_meta_to_text, generate_timestamp, read_yaml_file, save_query_results


def parse_arguments() -> Dict[str, Any]:
    """Parse command line arguments for the 'scrape' or 'chat' commands.

    -scrape: This command fetches and prepares data from the specified website using the starting URL. 
    It then initializes a chatbot that utilizes this data.

    -chat: This command initializes the chatbot immediately.
    It starts the chatbot using the data that was last scraped and prepared.

    Returns:
        dict: A dictionary containing the command and its associated arguments.
    """
    parser = argparse.ArgumentParser(description='SiteSavant - A chatbot for any website on-demand')
    subparsers = parser.add_subparsers(dest='command', required=True, help='commands')
    
    parser_scrape = subparsers.add_parser('scrape', help='Scrape a website and initialize the chatbot on the scraped data')
    parser_scrape.add_argument('url', help='The start URL for scraping', type=str)
    
    parser_chat = subparsers.add_parser('chat', help='Initialize the chatbot on  previously scraped data')

    args = parser.parse_args()
    return vars(args)

def handle_data_preparation(config: Dict[str, Any], file_paths: Dict[str, Any], timestamp: str) -> None:
    """Handles the entire data preparation pipeline, from scraping to indexing embeddings.

    Args:
        config (dict): The configuration settings for data preparation.
        file_paths (dict): The file paths for saving processed data.
        timestamp (str): A timestamp string to append to files for uniqueness.
    """
    # Scraping the website
    scraped_data = scrape_website(**config['website_scraper'])
    write_json_file(
        data=scraped_data,
        file_path=file_paths['website_scraper']['output_scraping_file_path'],
        timestamp=timestamp
        )

    # Creating embeddings
    modified_scraped_data = prepend_title_and_meta_to_text(scraped_data)
    cohere_api_key = os.getenv('COHERE_API_KEY') 
    embeddings_with_metadata = create_embeddings(
        scraped_data=modified_scraped_data,
        **config['embeddings_creator'], 
        cohere_api_key=cohere_api_key
        )
    save_embeddings_and_metadata(
        data=embeddings_with_metadata,
        data_dir=file_paths['embeddings_creator']['output_embeddings_processed_data_dir'],
        timestamp=timestamp
        )
    
    # Deduplicating embeddings
    unique_records, duplicate_records = deduplicate_embeddings(
    records=embeddings_with_metadata,
    **config['embeddings_deduplicator'],
    )
    sorted_duplicate_records = process_and_sort_duplicates(duplicate_records)
    write_json_file(
        data=sorted_duplicate_records,
        file_path=file_paths['embeddings_deduplicator']['output_duplicate_records_file_path'],
        timestamp=timestamp
        )
    save_embeddings_and_metadata(
        data=unique_records,
        data_dir=file_paths['embeddings_deduplicator']['output_embeddings_deduplicated_data_dir'],
        timestamp=timestamp
        )

    # Indexing embeddings
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    index_records(
        embeddings_data=unique_records,
        **config['embeddings_indexer'],
        pinecone_api_key=pinecone_api_key,
        )


def handle_chatbot_interaction(config: Dict[str, Any], file_paths: Dict[str, Any], timestamp: str) -> None:
    """Handle chatbot interactions processing user queries.

    Args:
        config (dict): The configuration for chatbot interaction.
        file_paths (dict): The file paths for saving chatbot interaction data.
        timestamp (str): A timestamp string to append to files for uniqueness.
    TODO: Get rid of changes to the logging level in this function.
    """
    logging.info("SiteSavant is ready to assist you. Enter your question or type 'quit' to exit: ")
    logging.getLogger().setLevel(logging.WARNING)
    
    while True:
        thinking_emoji = "\U0001F914"
        user_query = input(f"{thinking_emoji}: ").strip()

        if user_query.lower() == 'quit':
            logging.getLogger().setLevel(logging.INFO)
            logging.info("Session ended by the user.")
            break

        config['query_handler']['user_query'] = user_query
        config['chatbot_interactor']['user_query'] = user_query
        process_and_respond(config=config, file_paths=file_paths, timestamp=timestamp)


def process_and_respond(config: Dict[str, Any], file_paths: Dict[str, Any], timestamp: str) -> None:
    """Processes a user query and generates a chatbot response.

    Args:
        config (dict): The configuration for processing and responding to queries.
        file_paths (dict): The file paths for saving query results.
        timestamp (str): A timestamp string to append to files for uniqueness.
    """
    # Retrieve API keys
    cohere_api_key = os.getenv('COHERE_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')

    # Process the query
    raw_query_result = process_query(
        **config['query_handler'],
        cohere_api_key=cohere_api_key,
        pinecone_api_key=pinecone_api_key,
    )
    save_query_results(
        results=raw_query_result,
        file_path=file_paths['query_handler']['output_query_results_file_path'],
        timestamp=timestamp
    )

    # Generate chatbot interaction
    query_result = parse_query_results(raw_query_result)
    chat_response = generate_chat_response(
        query_results=query_result,
        **config['chatbot_interactor']
    )
    robot_emoji = "\U0001F916"
    print(f"{robot_emoji}: {chat_response}")


def main():
    """
    Orchestrates the setup and operation of the SiteSavant chatbot application.
    This is the main entry point for the application, handling the entire lifecycle of chatbot interactions based on website data.

    Steps:
    1. Initializes logging and loads environment variables.
    2. Generates a unique timestamp for tagging data and output files.
    3. Reads application configurations from 'config/parameters.yml' and from command-line arguments.
    4. Executes the data preparation phase, including web scraping and embeddings processing, unless skipped by the user.
    5. Enters a loop for chatbot interaction, processing and responding to user queries until termination.
    """
    logging.basicConfig(level=logging.INFO)
    setup_global_logger()
    load_dotenv()   
    timestamp = generate_timestamp()

    all_parameters = read_yaml_file('config/parameters.yml')
    config = all_parameters['main_config']
    file_paths = all_parameters['file_paths']
    cli_args = parse_arguments()

    if cli_args['command'] == 'scrape':
        config['website_scraper']['start_urls'] = cli_args['url']
        config['chatbot_interactor']['url'] = cli_args['url']
        handle_data_preparation(config=config, file_paths=file_paths, timestamp=timestamp)
        handle_chatbot_interaction(config=config, file_paths=file_paths, timestamp=timestamp)
    elif cli_args['command'] == 'chat':
        handle_chatbot_interaction(config=config, file_paths=file_paths, timestamp=timestamp)

if __name__ == '__main__':
    main()