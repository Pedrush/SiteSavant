import logging
import os
import pinecone
import requests
import datetime
from dotenv import load_dotenv
from config.logging_config import setup_global_logger
from services.embeddings_creator import CohereTextProcessingService
from utils.utils import read_yaml_file
from typing import Dict, Any


# TODO: Consider moving this function to a separate file
def save_query_results(results: Dict[str, Any], relative_path: str, timestamp: str = None) -> None:
    """
    Writes query results to a Markdown file at the specified path, optionally appending a timestamp to the filename,
    and logs the action. The path is assumed to be relative to the script's current working directory.

    Parameters:
    - results (Dict[str, Any]): The results from the query to be written to the file.
    - relative_path (str): The relative directory and base filename where the query results will be saved.
                           The directory must exist or will be created.
    - timestamp (str, optional): A string representing the timestamp to append to the filename.
                                 If provided, it will be appended before the file extension.

    Returns:
    None.
    """
    # Split the path into directory, base filename, and extension
    dir_path, filename = os.path.split(relative_path)
    base_filename, extension = os.path.splitext(filename)

    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)
    
    # Construct filename with timestamp
    if timestamp:
        filename_with_timestamp = f"{base_filename}_{timestamp}{extension}"
    else:
        filename_with_timestamp = f"{base_filename}{extension}"
    
    # Construct full path for the file with timestamp
    full_path_with_timestamp = os.path.join(dir_path, filename_with_timestamp)

    with open(full_path_with_timestamp, 'w') as file:
        file.write('# Query Results\n\n')
        for match in results.get('matches', []):  # Safe access to 'matches'
            file.write(f"## Score: {match['score']:.2f}\n")
            file.write(f"- **Text**: {match['metadata']['detokenized_chunk']}\n\n")

    logging.info(f"Results have been written to {full_path_with_timestamp}")

def parse_query_results(query_results: Dict[str, Any]) -> str:
    """
    Parses query results to extract and concatenate the 'detokenized_chunk' text from each match.

    Parameters:
    - query_results (Dict[str, Any]): The query results containing matches.

    Returns:
    str: A single string containing all 'detokenized_chunk' texts concatenated together, separated by '\n\n'.
    """
    # Initialize an empty list to store detokenized chunks
    detokenized_chunks = []

    # Iterate over each match to extract the 'detokenized_chunk'
    for match in query_results.get('matches', []):  # Safe access to 'matches'
        detokenized_chunk = match.get('metadata', {}).get('detokenized_chunk', '')
        if detokenized_chunk:  # Ensure the chunk is not empty
            detokenized_chunks.append(detokenized_chunk)

    # Join all detokenized chunks with '\n\n' separator
    concatenated_text = '\n\n'.join(detokenized_chunks)

    return concatenated_text

def process_query(user_query: str, model_name: str, pinecone_environment: str, index_name: str, 
                  top_k: int, cohere_api_key: str, pinecone_api_key: str) -> Dict[str, Any]:
    """
    Embeds a query using Cohere's text embedding service and queries a Pinecone index with the embedded query.

    Parameters:
    - user_query (str): The text query to be embedded and searched.
    - model_name (str): Name of the model to use for generating embeddings in Cohere.
    - pinecone_environment (str): Environment setting for Pinecone initialization.
    - index_name (str): Name of the Pinecone index to query.
    - top_k (int): Number of top results to retrieve from the Pinecone query.
    - cohere_api_key (str): API key for authenticating with the Cohere service.
    - pinecone_api_key (str): API key for authenticating with Pinecone.


    Returns:
    Dict[str, Any]: The result from the Pinecone query.
    """
    with requests.Session() as session:
        session.headers.update({
            'Authorization': f'Bearer {cohere_api_key}',
            'Content-Type': 'application/json'
        })
        cohere_service = CohereTextProcessingService(session)
        query_embedding = cohere_service.get_embedding(user_query, model_name=model_name, embedding_type='search_query')
    
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    index = pinecone.Index(index_name)
    result = index.query(query_embedding, top_k=top_k, include_metadata=True)

    return result

def main():
    # TODO: Write the docstring as in respectful_scraper.py

    logging.basicConfig(level=logging.INFO)
    setup_global_logger() 
    load_dotenv()

    # Configuration parameters
    config = read_yaml_file('config/parameters.yml')
    query_handler_config = config['query_handler_config']

    pinecone_environment = query_handler_config.get('pinecone_environment')
    index_name = query_handler_config.get('pinecone_index_name')
    model_name = query_handler_config.get('model_name')
    query = query_handler_config.get('query')
    top_k = query_handler_config.get('how_many_results_to_return')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    cohere_api_key = os.getenv('COHERE_API_KEY')

    # Extract the first three words from the query for the filename
    query_for_filename = '_'.join(query.split()[:3])

    # Embed the query
    with requests.Session() as session:
        session.headers.update({
            'Authorization': f'Bearer {cohere_api_key}',
            'Content-Type': 'application/json'
        })
        cohere_service = CohereTextProcessingService(session)
        query_handler = cohere_service.get_embedding(query, model_name=model_name, embedding_type='search_query')
    
    # Connect to Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    index = pinecone.Index(index_name)

    # Query Pinecone index
    result = index.query(query_handler, top_k=top_k, include_metadata=True)

    # Format the date and time in a suitable format for a filename
    # TODO: specify path: data/query_results as a parameter in parameters.yml
    now = datetime.datetime.now()
    filename = now.strftime(f"/root/repos/SiteSavant/data/query_results/{query_for_filename}_%Y-%m-%d_%H-%M.md")

    # Open a Markdown file to write the results
    with open(filename, 'w') as file:
        # Write a header for the file
        file.write(f'# Query Results: "{query}"\n\n')
        
        # Write each match to the file 
        for match in result['matches']:
            file.write(f"## Score: {match['score']:.2f}\n")
            file.write(f"- **Text**: {match['metadata']['detokenized_chunk']}\n\n")

    logging.info(f"Results have been written to {filename}")

if __name__ == "__main__":
    main()
