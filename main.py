import logging
import argparse
from services.website_scraper import scrape_website
from services.embeddings_creator import create_embeddings
from services.embeddings_deduplicator import deduplicate_embeddings, check_join_success, process_and_sort_duplicates
from services.embeddings_indexer import index_records
from services.query_handler import process_query, save_query_results, parse_query_results

from services.chatbot_interactor import generate_chat_response
from utils.utils import write_json_file, save_embeddings_and_metadata
import yaml
from config.logging_config import setup_global_logger
from dotenv import load_dotenv
import os
import datetime

def generate_timestamp():
    """Generate a timestamp string."""
    return datetime.datetime.now().strftime("%d-%m-%Y_%H_%M_%S")

def load_config(config_file):
    """Load and return the configuration from a YAML file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def parse_arguments():
    """Parse command line arguments for key parameters."""
    parser = argparse.ArgumentParser(description='SiteSavant - A chatbot for any website on demand')
    parser.add_argument('--user-query', help='Set the user query for the chatbot', type=str, required=True)
    parser.add_argument('--start-url', help='Override the start URL for scraping', type=str)
    parser.add_argument('--chatbot-model', help='Override the chatbot model name', type=str)
    parser.add_argument('--skip-data-preparation', help='Skip the data preparation steps and directly process the user query', action='store_true')
    return vars(parser.parse_args())

def merge_configurations(main_config, cli_args):
    """
    Merge command-line arguments with file configuration for key parameters.
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

def main():
    logging.basicConfig(level=logging.INFO)
    setup_global_logger()
    load_dotenv()   
    timestamp = generate_timestamp()

    all_parameters = load_config('config/parameters.yml')
    main_config = all_parameters['main_config']
    file_paths = all_parameters['file_paths']
    # cli_args = parse_arguments()
    cli_args = {
        'user_query': "How long will the gym be left open?",
        'skip_data_preparation': False,
        'start_url': None,
        'chatbot_model': None
    }
    config = merge_configurations(main_config, cli_args)

    # TODO: Enable to just run the chatbot interaction without the whole data preparation pipeline
    # TODO: Ensure that all write functions have timestamp as an argument
    if not cli_args['skip_data_preparation']:
        # Scraping website
        scraping_result = scrape_website(**config['website_scraper'])
        write_json_file(
            data=scraping_result['scraped_data'],
            output_file_path=file_paths['website_scraper']['scraping_output_dir'],
            timestamp=timestamp
            )


        # Creating embeddings
        cohere_api_key = os.getenv('COHERE_API_KEY') 
        embeddings_with_metadata = create_embeddings(
            scraped_data=scraping_result['scraped_data'],
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
        check_join_success(processed_data=unique_records, original_data=embeddings_with_metadata)
        sorted_duplicate_records = process_and_sort_duplicates(duplicate_records)
        write_json_file(
            sorted_duplicate_records,
            file_paths['embeddings_deduplicator']['output_duplicate_records'],
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
            pinecone_api_key=pinecone_api_key,
            embeddings_data=unique_records,
            **config['embeddings_indexer'],
            )


    # Query handling
    cohere_api_key = os.getenv('COHERE_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    raw_query_result = process_query(
        **config['query_handler'],
        cohere_api_key=cohere_api_key,
        pinecone_api_key=pinecone_api_key,
        )
    save_query_results(
        results=raw_query_result,
        relative_path=file_paths['query_handler']['output_query_results'],
        timestamp=timestamp
        )


    # Chatbot interaction
    query_result = parse_query_results(raw_query_result)
    chat_response = generate_chat_response(
        query_results=query_result,
        **config['chatbot_interactor']
        )
    logging.info(f"Chatbot response: {chat_response}")



if __name__ == '__main__':
    main()

    # python main.py --user-query="How long will the gym be left open?" --skip-data-preparation=True