import logging
import os
import pinecone
import requests
import datetime
from dotenv import load_dotenv
from config.logging_config import setup_global_logger
from services.creating_embeddings import CohereTextProcessingService
from utils.utils import read_yaml_file

def main():
    # TODO: Write the docstring as in respectful_scraper.py

    logging.basicConfig(level=logging.INFO)
    setup_global_logger() 
    load_dotenv()

    # Configuration parameters
    config = read_yaml_file('config/parameters.yml')
    querying_embeddings = config['querying_embeddings']

    pinecone_environment = querying_embeddings.get('pinecone_environment')
    index_name = querying_embeddings.get('pinecone_index_name')
    model_name = querying_embeddings.get('model_name')
    query = querying_embeddings.get('query')
    top_k = querying_embeddings.get('how_many_results_to_return')
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
        query_embeddings = cohere_service.get_embeddings(query, model_name=model_name, embeddings_type='search_query')
    
    # Connect to Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    index = pinecone.Index(index_name)

    # Query Pinecone index
    result = index.query(query_embeddings, top_k=top_k, include_metadata=True)

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
