"""
SiteSavant Main Application Orchestrator

This script acts as the central orchestrator for the SiteSavant application, handling the entire lifecycle of chatbot interactions
based on website data. It integrates various services for web scraping, data processing, embeddings creation and deduplication,
indexing, and chatbot interactions. The application supports custom configurations and command-line arguments to tailor
the processing and interaction flow.

CLI Arguments:
- --user-query: Required. Set the initial user query for the chatbot to process.
- --start-url: Optional. Override the default start URL for scraping with a specified URL.
- --chatbot-model: Optional. Override the default chatbot model name.
- --skip-data-preparation: Optional. Skip the data preparation steps and directly
    process the user query based on already indexed data.

Key Features:
- Web scraping from specified URLs.
- Integration with external APIs for embeddings and data indexing.
- A responsive chatbot that processes user queries against scraped data.

Usage:
Run the script with necessary command-line arguments to specify user queries, start URLs, and other options.
Use environment variables for sensitive API keys and configurations.

Example:
python main.py --user-query="Who is the CEO of the company?" --start-url="https://example.com"
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
    """Parse command line arguments for key parameters.

    Returns:
        dict: A dictionary containing all parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='SiteSavant - A chatbot for any website on demand')
    parser.add_argument('--user-query', help='Set the user query for the chatbot', type=str, required=True)
    parser.add_argument('--start-url', help='Override the start URL for scraping', type=str)
    parser.add_argument('--chatbot-model', help='Override the chatbot model name', type=str)
    parser.add_argument('--skip-data-preparation', help='Skip the data preparation steps and directly process the user query', action='store_true')
    return vars(parser.parse_args())


def merge_configurations(main_config: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """Merge command-line arguments with file configuration for key parameters.

    Args:
        main_config (dict): The main configuration loaded from a file.
        cli_args (dict): Command-line arguments provided by the user.

    Returns:
        dict: The updated configuration dictionary after merging.
    """
    if cli_args['start_url']:
        main_config['website_scraper']['start_url'] = cli_args['start_url']
        # TODO: consider removing the url argument in chatbot_interactor
        main_config['chatbot_interactor']['url'] = cli_args['start_url']
    if cli_args['chatbot_model']:
        main_config['chatbot_interactor']['model_name'] = cli_args['chatbot_model']
    if cli_args['user_query']:
        main_config['query_handler']['user_query'] = cli_args['user_query']
        main_config['chatbot_interactor']['user_query'] = cli_args['user_query']
    return main_config


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
        file_path=file_paths['website_scraper']['scraping_output_file_path'],
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
    """Handle chatbot interactions, processing initial and subsequent user queries.

    Args:
        config (dict): The configuration for chatbot interaction.
        file_paths (dict): The file paths for saving chatbot interaction data.
        timestamp (str): A timestamp string to append to files for uniqueness.
    """
    # Process the initial user query from the config
    initial_query = config['query_handler']['user_query']
    logging.info(f"Processing query: {initial_query}")
    process_and_respond(config=config, file_paths=file_paths, timestamp=timestamp)

    # Loop for subsequent user interactions
    while True:
        user_query = input("Enter your question or type 'quit' to exit: ").strip()
        if user_query.lower() == 'quit':
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
    logging.info(f"Chatbot response: {chat_response}")


def main():
    """
    Orchestrates the setup and operation of the SiteSavant chatbot application.

    Steps:
    1. Initializes logging and loads environment variables.
    2. Generates a unique timestamp for tagging data and output files.
    3. Reads application configurations from 'config/parameters.yml' and merges them with command-line arguments.
    4. Executes the data preparation phase, including web scraping and embeddings processing, unless skipped by the user.
    5. Enters a loop for chatbot interaction, processing and responding to user queries until termination.
    """
    logging.basicConfig(level=logging.INFO)
    setup_global_logger()
    load_dotenv()   
    timestamp = generate_timestamp()

    all_parameters = read_yaml_file('config/parameters.yml')
    main_config = all_parameters['main_config']
    file_paths = all_parameters['file_paths']
    # cli_args = parse_arguments()
    cli_args = {
        'user_query': "What is the CEO of the company?",
        'skip_data_preparation': False,
        'start_url': None,
        'chatbot_model': None
    }
    config = merge_configurations(main_config, cli_args)

    if not cli_args['skip_data_preparation']:
        handle_data_preparation(config=config, file_paths=file_paths, timestamp=timestamp)
    
    handle_chatbot_interaction(config=config, file_paths=file_paths, timestamp=timestamp)


if __name__ == '__main__':
    main()