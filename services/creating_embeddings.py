import logging
import os
import requests
import time
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List
from dotenv import load_dotenv
from config.logging_config import setup_global_logger
from utils.utils import read_json_file, write_json_file

# TODO: JSON file with embeddings shouldn't repeat the whole text over and over again

class TextProcessingService(ABC):
    @abstractmethod
    def tokenize_text(self, text: str) -> List[int]:
        """
        Tokenizes the given text into a list of tokens.
        """
        pass

    @abstractmethod
    def get_embeddings(self, tokens: List[str]) -> List[List[float]]:
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
    def __init__(self, session: requests.Session, model_name: str = 'embed-multilingual-v2.0', max_embedding_model_input_length: int = 512):
        self.session = session
        self.model_name = model_name
        self.max_embedding_model_input_length = max_embedding_model_input_length


    def tokenize_text(self, text: str, model_name: str = 'embed-multilingual-v2.0') -> List[int]:
        """
        Tokenizes a text using the Cohere API. Implemented for precise control
        over token-wise text chunking to optimize embeddings quality.

        Args:
            text (str): The text to tokenize.
            model_name (str, optional): The model name compatible with the tokenizer. If None, uses the model
                                        set during class instantiation. Defaults to None.

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

        time.sleep(1.2)
        # logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        response = self.session.post(url, json=data)

        if response.status_code == 200:
            return response.json().get('text', '')
        else:
            raise Exception(f"Error detokenizing text: {response.text}")

    def get_embeddings(self, text: str, input_type: str = 'search_document') -> List[float]:
        """
        Retrieves embeddings for the text using the Cohere API.

        Args:
            tokens (str): A string to embed.
            input_type (str): Specifies the type of input you're giving to the model.
            Defaults to 'search_document' to embed strings that can be later searched over.
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
        data = {
            'texts': [text],
            'model': self.model_name,
            'input_type': input_type,
        }
        response = self.session.post(url, json=data)
        if response.status_code == 200:
            embeddings = response.json().get('embeddings', [])[0]
        else:
            raise Exception(f"Error getting embeddings: {response.text}")
        return embeddings


def chunk_tokens(tokens: List[int], max_size: int) -> List[List[int]]:
    """
    Splits a list of tokens into chunks of a specified maximum size.

    Args:
        tokens (List[int]): The list of tokens to be chunked.
        max_size (int): The maximum size of each chunk.

    Returns:
        List[List[int]]: A list of token chunks.
    """
    return [tokens[i:i + max_size] for i in range(0, len(tokens), max_size)]

def embed_file_contents(file_path: str, text_processor: TextProcessingService):
    """
    Processes a single file by tokenizing, detokenizing, and obtaining embeddings for the text.

    Args:
        file_path (str): The path of the file to process.
        text_processor (TextProcessingService): The text processing service to use.

    Returns:
        List[dict]: A list of processed data records.
    """
    logging.info(f"Processing file: {file_path}")
    try:
        json_data = read_json_file(file_path)
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return

    processed_data = []
    for record in tqdm(json_data, desc="Processing records"):
        text = record.get("text", "")
        if text:
            try:
                tokens = text_processor.tokenize_text(text)
                token_chunks = chunk_tokens(tokens, max_embedding_model_input_length)
                for chunk in tqdm(token_chunks, desc="Processing token chunks", leave=False):
                    detokenized_chunk = text_processor.detokenize_text(chunk)
                    embeddings = text_processor.get_embeddings(detokenized_chunk)
                    processed_record = {
                        **record,  # Include original record data
                        'tokenized_chunk': chunk,
                        'detokenized_chunk': detokenized_chunk,
                        'embeddings': embeddings
                    }
                    processed_data.append(processed_record)
            except Exception as e:
                logging.error(f"Error processing text: {e}")

    return processed_data



def main(file_path: str, text_processor: TextProcessingService, processed_data_dir: str):
    processed_data = embed_file_contents(file_path, text_processor)
    # TODO: write processed data to a file. It's better to store embeddings in, for example, a pickle/parquet file. JSON might hold references to these files inside of it. JSON is for the text data primarily, not long floats.
    if processed_data:
        output_file_name = os.path.basename(file_path).replace('.json', '_processed.json')
        output_file_path = os.path.join(processed_data_dir, output_file_name)
        write_json_file(processed_data, output_file_path)
    
    return processed_data

if __name__ == "__main__":
    # TODO: move all this stuff to the actual main function.
    # Before running the script, make sure to set the following variables:
    # model_name 
    # input_type for embeddings (search_document, search_query, classification, clustering)
    # scraped_data_file_path


    logging.basicConfig(level=logging.INFO)
    setup_global_logger() 
    load_dotenv()

    max_embedding_model_input_length = 512 # optimal size for optimizing embeddings quality as per https://docs.cohere.com/reference/embed
    processed_data_dir = "data/processed_data"

    cohere_api_key = os.getenv('COHERE_API_KEY')
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY not found in the environment variables.")

    scraped_data_file_path = os.getenv('SCRAPED_DATA_FILE_PATH')
    if not scraped_data_file_path:
        raise ValueError("SCRAPED_DATA_FILE_PATH not found in the environment variables.")

    #TODO: Consider whether to use a context manager from within the class
    #     @contextmanager
    # def session(self):
    #     with requests.Session() as session:
    #         yield session

    with requests.Session() as session:
        session.headers.update({
            'Authorization': f'Bearer {cohere_api_key}',
            'Content-Type': 'application/json'
        })
        cohere_service = CohereTextProcessingService(session, max_embedding_model_input_length=max_embedding_model_input_length)
        main(scraped_data_file_path, cohere_service, processed_data_dir)