import logging
import os
import requests
import time
import re
import h5py
import json
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dotenv import load_dotenv
from config.logging_config import setup_global_logger
from utils.utils import read_json_file, write_json_file, read_yaml_file, save_embeddings_and_metadata
from datetime import datetime

# TODO: JSON file with embeddings shouldn't repeat the whole text over and over again

class TextProcessingService(ABC):
    @abstractmethod
    def tokenize_text(self, text: str) -> List[int]:
        """
        Tokenizes the given text into a list of tokens.
        """
        pass

    @abstractmethod
    def get_embedding(self, tokens: List[str]) -> List[List[float]]:
        """
        Retrieves embeddings for the given text.
        """
        pass

    @abstractmethod
    def detokenize_text(self, tokens: List[int], model_name: str = None) -> str:
        """
        Detokenizes the given list of token IDs to a string of text.
        """
        pass

class CohereTextProcessingService(TextProcessingService):
    def __init__(
            self, 
            session: requests.Session, 
            model_name: str = 'embed-multilingual-v2.0', 
            max_embedding_model_input_length: int = 512,
            embedding_type: str = 'search_document', 
            ):
        
        self.session = session
        self.model_name = model_name
        self.max_embedding_model_input_length = max_embedding_model_input_length
        self.embedding_type = embedding_type

    # TODO: Consider whether to use a context manager from within the class
    #     @contextmanager
    # def session(self):
    #     with requests.Session() as session:
    #         yield session

    def tokenize_text(self, text: str, model_name: str = None) -> List[int]:
        """
        Tokenizes a text using the Cohere API. Implemented for precise control
        over token-wise text chunking to optimize embeddings quality.

        Args:
            text (str): The text to tokenize.
            model_name (str, optional): The model name compatible with the tokenizer. If None, uses the model
                                        set during class instantiation.

        Returns:
            List[int]: A list of tokens.

        Raises:
            ValueError: If the text length exceeds the maximum limit.
            Exception: If there is an error in the tokenization process.
        TODO: gracefully continue if the text length exceeds the maximum limit. Just log an error, inform about truncating, and continue.
        """
        max_length = 65536
        if len(text) > max_length:
            logging.warning(f"Text length exceeds the maximum limit of {max_length} characters. The cohere API doesn't handle more during tokenizetion. Text was therefore truncated to 65534 characters to meet the limit.")
            text = text[0:65534]

        selected_model = model_name if model_name else self.model_name

        url = 'https://api.cohere.ai/v1/tokenize'
        data = {'text': text, 'model': selected_model}
        response = self.session.post(url, json=data)
        
        if response.status_code == 200:
            return response.json().get('tokens', [])
        else:
            raise Exception(f"Error tokenizing text: {response.text}")
        
    def detokenize_text(self, tokens: List[int], model_name: str = None) -> str:
        """
        Detokenize a list of tokens using the Cohere API.

        Args:
            tokens (List[int]): The list of tokens to be detokenized.
            model_name (str, optional): The model name compatible with the detokenizer. If None, uses the model
                                        set during class instantiation. Defaults to None.

        Returns:
            str: The detokenized text.

        Raises:
            Exception: If there is an error in the detokenization process.
        """
        url = 'https://api.cohere.ai/v1/detokenize'
        selected_model = model_name if model_name else self.model_name
        data = {'tokens': tokens, 'model': selected_model}
        response = self.session.post(url, json=data)

        if response.status_code == 200:
            return response.json().get('text', '')
        else:
            raise Exception(f"Error detokenizing text: {response.text}")

    def get_embedding(self, text: str, model_name: str = None, embedding_type: str = None) -> List[float]:
        """
        Retrieves embedding for the text using the Cohere API.

        Args:
            tokens (str): A string to embed.
            model_name (str, optional): The model name compatible with the detokenizer. If None, uses the model
            set during class instantiation. Defaults to None.
            embedding_type (str): Specifies the type of the embeddings. Defaults to None.
            Can be:
            -'search_document', 
            -'search_query', 
            -'classification',
            -'clustering'.

        Returns:
            List[float]: A list of embeddings.

        Raises:
            Exception: If there is an error in retrieving embeddings.
        """
        url = 'https://api.cohere.ai/v1/embed'
        selected_model = model_name if model_name else self.model_name
        input_type = embedding_type if embedding_type else self.embedding_type
        data = {
            'texts': [text],
            'model': selected_model,
            'input_type': input_type,
        }
        response = self.session.post(url, json=data)
        if response.status_code == 200:
            embedding = response.json().get('embeddings', [])[0]
        else:
            raise Exception(f"Error getting embeddings: {response.text}")
        return embedding


def chunk_tokens(tokens: List[int], max_size: int, min_size: int = 0) -> List[List[int]]:
    """
    Splits a list of tokens into chunks with specified maximum and minimum sizes.

    Args:
        tokens (List[int]): The list of tokens to be chunked.
        max_size (int): The maximum size of each chunk.
        min_size (int): The minimum size for the last chunk. Defaults to 0.

    Returns:
        List[List[int]]: A list of token chunks that meet the size constraints.
    """
    chunks = [tokens[i:i + max_size] for i in range(0, len(tokens), max_size)]

    if chunks and len(chunks[-1]) < min_size:
        chunks.pop()
        logging.info(f"Last chunk of tokens was too short and was removed. Minimum chunk length: {min_size}")

    return chunks

def embed_file_contents(json_data: dict, text_processor: TextProcessingService, max_embedding_model_input_length: int = 512) -> List[dict]:
    """
    Processes a single file by tokenizing, detokenizing, and obtaining embeddings for the text.

    Args:
        json_data (dict): A file to process.
        text_processor (TextProcessingService): The text processing service to use.

    Returns:
        List[dict]: A list of processed data records.
    """

    processed_data = []
    for record in tqdm(json_data, desc="Processing records"):
        text = record.get("text", "")
        if text:
            try:
                tokens = text_processor.tokenize_text(text)
                token_chunks = chunk_tokens(tokens, max_size=max_embedding_model_input_length, min_size=10)
                for chunk in tqdm(token_chunks, desc="Processing token chunks", leave=False):
                    detokenized_chunk = text_processor.detokenize_text(chunk)
                    embedding = text_processor.get_embedding(detokenized_chunk)
                    processed_record = {
                        **record,  # Include original record data
                        'tokenized_chunk': chunk,
                        'detokenized_chunk': detokenized_chunk,
                        'embedding': embedding
                    }
                    processed_data.append(processed_record)
            except Exception as e:
                logging.error(f"Error processing text: {e}")

    return processed_data

def create_embeddings(scraped_data, max_embedding_model_input_length, model_name, embedding_type, cohere_api_key):
    """
    Create embeddings from scraped data using the specified embedding model.
    Note: The function is tightly coupled with CohereTextProcessingService.

    Args:
    scraped_data (dict): The scraped data text for which embeddings are to be created.
    model_name (str): The name of the Cohere model to use for embeddings.
    max_embedding_model_input_length (int): Maximum length of the input to the embedding model in tokens.
    embedding_type (str): The type of embeddings to use.
    cohere_api_key (str): The API key for accessing Cohere's services.

    Returns:
    dict: Processed data containing embeddings and their metadata.
    TODO:
    - Decouple the function from CohereTextProcessingService
    """
    
    with requests.Session() as session:
        session.headers.update({
            'Authorization': f'Bearer {cohere_api_key}',
            'Content-Type': 'application/json'
        })

        cohere_service = CohereTextProcessingService(
            session,
            model_name=model_name,
            max_embedding_model_input_length=max_embedding_model_input_length,
            embedding_type=embedding_type,
            )

        # Process the file and handle the data
        processed_data = embed_file_contents(
            json_data=scraped_data,
            text_processor=cohere_service,
            max_embedding_model_input_length=max_embedding_model_input_length,
        )

    return processed_data
    

def main():
    # TODO: Write the docstring as in respectful_scraper.py
    # Initialize logging and load environment variables
    logging.basicConfig(level=logging.INFO)
    setup_global_logger() 
    load_dotenv()

    # Configuration parameters
    config = read_yaml_file('config/parameters.yml')
    embeddings_creator_config = config['embeddings_creator']

    scraped_data_file_path = embeddings_creator_config.get('input_scraped_data_file_path') # File path of the scraped data
    max_embedding_model_input_length = embeddings_creator_config.get('max_embedding_model_input_length') # Maximum length of the input to the embedding model as measured in tokens
    processed_data_dir = embeddings_creator_config.get('output_embedding_processed_data_dir') # Directory where the processed data will be saved
    model_name = embeddings_creator_config.get('embedding_model_name') # Name of the embedding model to use
    embedding_type = embeddings_creator_config.get('embedding_type') # Type of embeddings to use
    cohere_api_key = os.getenv('COHERE_API_KEY') # Cohere API key

    logging.info(f"Processing file: {scraped_data_file_path}")
    
    try:
        json_data = read_json_file(scraped_data_file_path)
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return

    embeddings_with_metadata = create_embeddings(scraped_data=json_data, max_embedding_model_input_length=max_embedding_model_input_length, model_name=model_name, embedding_type=embedding_type, cohere_api_key=cohere_api_key, processed_data_dir=processed_data_dir)
    
    # Save the processed data
    if embeddings_with_metadata:
        save_embeddings_and_metadata(
            data=embeddings_with_metadata,
            data_dir=processed_data_dir,
            metadata_file_name='processed_metadata',
            embeddings_file_name='processed_embeddings_values'
            )
    else:
        logging.critical("No data to save.")


if __name__ == "__main__":
    main()