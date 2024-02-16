"""
This script is designed to integrate text processing and vector search capabilities,
leveraging Cohere's text embedding service and Pinecone's vector database for efficient query processing and retrieval. 
It aims to embed user queries into vector space using Cohere and search for the most
relevant results in a Pinecone index based on these embeddings.

Functional Overview:
- Initializes logging and configures settings through environment variables and YAML configuration files.
- Generates a timestamp for each operation, facilitating tracking and logging of query processing times.
- Loads configuration parameters and API keys for Cohere and Pinecone.
- Embeds user queries using Cohere's advanced text processing capabilities,
 converting natural language queries into vector representations suitable for similarity search.
- Queries a Pinecone index with the generated embeddings, retrieving the most relevant results based on vector similarity.
- Saves query results with timestamps, enabling organized storage and easy retrieval of query outcomes for further analysis or review.

Components:
- parse_query_results: Extracts and concatenates 'detokenized_chunk' texts from query matches, providing a coherent and readable summary of query results.
- process_query: Orchestrates the embedding of user queries via Cohere and subsequent querying of a Pinecone index, encapsulating the core functionality of text processing and vector search integration.

Usage:
Can be used as a standalone module. Additionally, the functions are designed to integrate with a larger processing pipeline, 
as demonstrated in the main orchestrator script (main.py) within this project.
"""

# Standard library imports
import datetime
import logging
import os
from typing import Any, Dict

# Related third-party imports
from dotenv import load_dotenv
import pinecone
import requests

# Local application/library specific imports
from config.logging_config import setup_global_logger
from services.embeddings_creator import CohereTextProcessingService
from utils.utils import read_yaml_file, save_query_results, generate_timestamp


def parse_query_results(query_results: Dict[str, Any]) -> str:
    """
    Parses query results to extract and concatenate the 'detokenized_chunk' text from each match.

    Parameters:
    - query_results (Dict[str, Any]): The query results containing matches.

    Returns:
    str: A single string containing all 'detokenized_chunk' texts concatenated together, separated by '\n\n'.
    """
    detokenized_chunks = []

    # Iterate over each match to extract the 'detokenized_chunk'
    for match in query_results.get('matches', []):
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
    """
    Demonstrates the capabilities of various functions for processing a text query. This function
    integrates with Cohere's text embedding service to embed the query and then uses Pinecone to
    search for the most relevant results based on the embedded query.

    The process involves the following steps:
    1. Initialize logging and load environment variables.
    2. Generate a timestamp for tracking when the query was processed.
    3. Load configuration parameters and file paths from a YAML file.
    4. Retrieve API keys for Cohere and Pinecone from environment variables.
    5. Embed the user query using Cohere's text embedding service.
    6. Query a Pinecone index with the obtained embedding to retrieve the most relevant results.
    7. Save the query results to a specified file path with a timestamp.
    """
    logging.basicConfig(level=logging.INFO)
    setup_global_logger()
    timestamp = generate_timestamp()
    load_dotenv()

    all_parameters = read_yaml_file('config/parameters.yml')
    config = all_parameters['main_config']
    file_paths = all_parameters['file_paths']

    # Process the query
    cohere_api_key = os.getenv('COHERE_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
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

if __name__ == "__main__":
    main()