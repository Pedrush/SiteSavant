"""
This script processes text data to generate embeddings using the Cohere API. It defines a
framework for text processing that includes tokenization, detokenization, and embedding generation,
and is structured to support different processing services through an abstract base class and a
concrete implementation for Cohere.

Functional Overview:
- Sets up logging and reads configuration from YAML files.
- Preprocesses text data by appending titles and metadata.
- Tokenizes text, manages token chunks to fit API limits, and generates embeddings for each chunk.
- Saves embeddings and metadata, organized by a generated timestamp.

Components:
- `TextProcessingService`: An abstract base class for text processing tasks.
- `CohereTextProcessingService`: Implements the abstract base class to use Cohere's API for text
processing.
- Utility functions for reading configurations, preprocessing text, and saving results.

Usage:
Can be used as a standalone module. Additionally, the functions are designed to integrate with
a larger processing pipeline, as demonstrated in the main orchestrator script (main.py) within
this project.

Note: Ensure a .env file containing necessary API keys and environment variables is present in the
project root for secure API access.
"""

# Standard library imports
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List

# Related third-party imports
from dotenv import load_dotenv
import requests
from tqdm import tqdm

# Local application/library specific imports
from config.logging_config import setup_global_logger
from utils.utils import (
    generate_timestamp,
    prepend_title_and_meta_to_text,
    read_json_file,
    read_yaml_file,
    save_embeddings_and_metadata,
)


class TextProcessingService(ABC):
    @abstractmethod
    def tokenize_text(self, text: str) -> List[int]:
        """Tokenizes the given text into a list of tokens."""
        pass

    @abstractmethod
    def get_embedding(self, tokens: List[str]) -> List[List[float]]:
        """Retrieves embeddings for the given text."""
        pass

    @abstractmethod
    def detokenize_text(self, tokens: List[int], model_name: str = None) -> str:
        """Detokenizes the given list of token IDs to a string of text."""
        pass


class CohereTextProcessingService(TextProcessingService):
    def __init__(
        self,
        session: requests.Session,
        model_name: str = "embed-multilingual-v2.0",
        embedding_type: str = "search_document",
    ):

        self.session = session
        self.model_name = model_name
        self.embedding_type = embedding_type

    # TODO: Consider whether to use a context manager from within the class
    #     @contextmanager
    # def session(self):
    #     with requests.Session() as session:
    #         yield session

    def tokenize_text(self, text: str, model_name: str = None) -> List[int]:
        """Tokenizes a text using the Cohere API. Implemented for precise control over token-wise
        text chunking to optimize embeddings quality.

        Args:
            text (str): The text to tokenize.
            model_name (str, optional): The model name compatible with the tokenizer. If None,
            uses the model set during class instantiation.

        Returns:
            List[int]: A list of tokens.

        Raises:
            ValueError: If the text length exceeds the maximum limit.
            Exception: If there is an error in the tokenization process.
        TODO: gracefully continue if the text length exceeds the maximum limit. Just log an error,
              inform about truncating, and continue.
        """
        max_length = 65536
        if len(text) > max_length:
            logging.warning(
                f"Text length exceeds the maximum limit of {max_length} characters. "
                "The cohere API doesn't handle more during tokenization. "
                "Text was therefore truncated to 65534 characters to meet the limit."
            )
            text = text[0:max_length]

        selected_model = model_name if model_name else self.model_name

        url = "https://api.cohere.ai/v1/tokenize"
        data = {"text": text, "model": selected_model}
        response = self.session.post(url, json=data)

        if response.status_code == 200:
            return response.json().get("tokens", [])
        else:
            raise Exception(f"Error tokenizing text: {response.text}")

    def detokenize_text(self, tokens: List[int], model_name: str = None) -> str:
        """
        Detokenize a list of tokens using the Cohere API. It's used precisely control the number
        of tokens in the input to the embedding model.

        Args:
            tokens (List[int]): The list of tokens to be detokenized.
            model_name (str, optional): The model name compatible with the detokenizer. If None,
            uses the model set during class instantiation. Defaults to None.

        Returns:
            str: The detokenized text.

        Raises:
            Exception: If there is an error in the detokenization process.
        """
        url = "https://api.cohere.ai/v1/detokenize"
        selected_model = model_name if model_name else self.model_name
        data = {"tokens": tokens, "model": selected_model}
        response = self.session.post(url, json=data)

        if response.status_code == 200:
            return response.json().get("text", "")
        else:
            raise Exception(f"Error detokenizing text: {response.text}")

    def get_embedding(
        self, text: str, model_name: str = None, embedding_type: str = None
    ) -> List[float]:
        """Retrieves embedding for the text using the Cohere API.

        Args:
            tokens (str): A string to embed.
            model_name (str, optional): The model name compatible with the detokenizer. If None,
            uses the model set during class instantiation. Defaults to None.
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
        url = "https://api.cohere.ai/v1/embed"
        selected_model = model_name if model_name else self.model_name
        input_type = embedding_type if embedding_type else self.embedding_type
        data = {
            "texts": [text],
            "model": selected_model,
            "input_type": input_type,
        }
        response = self.session.post(url, json=data)
        if response.status_code == 200:
            embedding = response.json().get("embeddings", [])[0]
        else:
            raise Exception(f"Error getting embeddings: {response.text}")
        return embedding


def chunk_tokens(tokens: List[int], max_size: int, min_size: int) -> List[List[int]]:
    """Splits a list of tokens into chunks with specified maximum and minimum sizes.

    Args:
        tokens (List[int]): The list of tokens to be chunked.
        max_size (int): The maximum size of each chunk.
        min_size (int): The minimum size for the last chunk.

    Returns:
        List[List[int]]: A list of token chunks that meet the size constraints.
    """
    chunks = [tokens[i : i + max_size] for i in range(0, len(tokens), max_size)]

    if chunks and len(chunks[-1]) < min_size:
        chunks.pop()
        logging.info(
            f"Last chunk of tokens was too short and was removed. Minimum chunk length: {min_size}"
        )

    return chunks


def embed_file_contents(
    json_data: dict,
    text_processor: TextProcessingService,
    max_embedding_model_input_length: int = 512,
    minimum_chunk_length_in_tokens: int = 10,
) -> List[dict]:
    """Processes a single file by tokenizing, detokenizing, and obtaining embeddings for the text.

    Args:
        json_data (dict): A file to process.
        text_processor (TextProcessingService): The text processing service to use.
        max_embedding_model_input_length (int): Maximum length of the input to the embedding model
        in tokens.
        min_size (int): The minimum size for the last chunk. Defaults to 10.

    Returns:
        List[dict]: A list of processed data records.
    """

    processed_data = []
    for i, record in tqdm(enumerate(json_data), desc="Processing records", total=len(json_data)):
        text = record.get("text", "")
        if text:
            try:
                tokens = text_processor.tokenize_text(text)
                token_chunks = chunk_tokens(
                    tokens,
                    max_size=max_embedding_model_input_length,
                    min_size=minimum_chunk_length_in_tokens,
                )
                for k, chunk in tqdm(
                    enumerate(token_chunks), desc="Processing token chunks", leave=False
                ):
                    detokenized_chunk = text_processor.detokenize_text(chunk)
                    embedding = text_processor.get_embedding(detokenized_chunk)
                    processed_record = {
                        **record,  # Include original record data
                        "tokenized_chunk": chunk,
                        "detokenized_chunk": detokenized_chunk,
                        "embedding": embedding,
                        "embedding_id": f"{i}_{k}",
                    }
                    processed_data.append(processed_record)
            except Exception as e:
                logging.error(f"Error processing text: {e}")

    return processed_data


def create_embeddings(
    scraped_data: Dict[str, str],
    embedding_model_name: str,
    embedding_type: str,
    max_embedding_model_input_length: int,
    minimum_chunk_length_in_tokens: int,
    cohere_api_key: str,
) -> Dict[str, any]:
    """
    Create embeddings from scraped data using the specified embedding model.
    Note: The function is tightly coupled with CohereTextProcessingService.

    Args:
        scraped_data (dict): The scraped data text for which embeddings are to be created.
        embedding_model_name (str): The name of the Cohere model to use for embeddings.
        embedding_type (str): The type of embeddings to use.
        max_embedding_model_input_length (int): Maximum length of the input to the embedding model
        in tokens.
        minimum_chunk_length_in_tokens (int): The minimum chunk length in tokens.
        cohere_api_key (str): The API key for accessing Cohere's services.


    Returns:
        dict: Processed data containing embeddings and their metadata.
    TODO: Decouple the function from CohereTextProcessingService
    TODO: JSON file with embeddings shouldn't repeat the whole text over and over again
    """

    with requests.Session() as session:
        session.headers.update(
            {"Authorization": f"Bearer {cohere_api_key}", "Content-Type": "application/json"}
        )

        cohere_service = CohereTextProcessingService(
            session,
            model_name=embedding_model_name,
            embedding_type=embedding_type,
        )

        # Process the file and handle the data
        processed_data = embed_file_contents(
            json_data=scraped_data,
            text_processor=cohere_service,
            max_embedding_model_input_length=max_embedding_model_input_length,
            minimum_chunk_length_in_tokens=minimum_chunk_length_in_tokens,
        )

    return processed_data


def main():
    """Demonstrates the capabilities of various functions for processing text and generating
    embeddings.

    This function serves as an entry point for a text processing and embedding generation pipeline.
    It performs the following steps:

    Steps:
    1. Sets up global logging configuration.
    2. Generates a timestamp for the current run.
    3. Reads configuration parameters and file paths from a YAML file.
    4. Reads scraped data from a JSON file and preprocesses it by prepending titles and metadata.
    5. Creates text embeddings using the Cohere API by:
        a. Tokenizing the text.
        b. Chunking tokens to fit within the API's maximum input length.
        c. Detokenizing tokens to reconstruct text chunks.
        d. Generating embeddings for each chunk.
    6. Saves the generated embeddings and metadata to a specified directory, organized by the
       generated timestamp.
    """
    # Config
    logging.basicConfig(level=logging.INFO)
    setup_global_logger()
    timestamp = generate_timestamp()
    load_dotenv()

    all_parameters = read_yaml_file("config/parameters.yml")
    config = all_parameters["main_config"]
    file_paths = all_parameters["file_paths"]

    # Creating embeddings
    scraped_data = read_json_file(file_paths["embeddings_creator"]["input_scraped_data_file_path"])
    modified_scraped_data = prepend_title_and_meta_to_text(scraped_data)
    cohere_api_key = os.getenv("COHERE_API_KEY")
    embeddings_with_metadata = create_embeddings(
        scraped_data=modified_scraped_data,
        **config["embeddings_creator"],
        cohere_api_key=cohere_api_key,
    )
    save_embeddings_and_metadata(
        data=embeddings_with_metadata,
        data_dir=file_paths["embeddings_creator"]["output_embeddings_processed_data_dir"],
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
