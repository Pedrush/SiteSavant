import logging
import os
import pinecone
import requests
from dotenv import load_dotenv
from config.logging_config import setup_global_logger
from services.creating_embeddings import CohereTextProcessingService
from services.indexing_embeddings import init_pinecone
import datetime


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_global_logger() 
    load_dotenv()

    cohere_api_key = os.getenv('COHERE_API_KEY')
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY not found in the environment variables.")

    query = "Who did aviva aquire for 100 milion pounds?"

    # Extract the first three words from the query for the filename
    query_for_filename = '_'.join(query.split()[:3])

    with requests.Session() as session:
        session.headers.update({
            'Authorization': f'Bearer {cohere_api_key}',
            'Content-Type': 'application/json'
        })
        cohere_service = CohereTextProcessingService(session)
        query_embeddings = cohere_service.get_embeddings(query, input_type='search_query')
    
    # Connect to Pinecone
    init_pinecone()
    index_name = os.getenv('PINECONE_INDEX_NAME')
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME not found in the environment variables.")
    index = pinecone.Index(index_name)

    # Query your index (assuming you have an index object)
    # Replace 'index' with your actual index object and its querying method
result = index.query(query_embeddings, top_k=5, include_metadata=True)

# Get the current date and time
now = datetime.datetime.now()

# Format the date and time in a suitable format for a filename
# Example format: 'query_results_2024-01-09_13-45.md'
filename = now.strftime(f"/root/repos/SiteSavant/data/query_results/{query_for_filename}_%Y-%m-%d_%H-%M.md")

# Open a Markdown file to write the results
with open(filename, 'w') as file:
    # Write a header for the file
    file.write("# Query Results\n\n")
    
    # Write each match to the file 
    for match in result['matches']:
        file.write(f"## Score: {match['score']:.2f}\n")
        file.write(f"- **Text**: {match['metadata']['detokenized_chunk']}\n\n")

logging.info(f"Results have been written to {filename}")
