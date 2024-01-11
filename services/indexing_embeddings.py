import logging
import os
import pinecone
from dotenv import load_dotenv
from config.logging_config import setup_global_logger
from utils.utils import read_json_file
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

def validate_embeddings(json_data: List[Dict[str, Any]]) -> None:
    """
    Validates the embeddings in the JSON data.

    Args:
        json_data (List[Dict[str, Any]]): A list of dictionaries containing embeddings.

    Raises:
        ValueError: If the data is empty, embeddings are not floats, or have inconsistent dimensions.

    Ensures that all embeddings have the same dimensions and are of the correct type (float).
    """
    if not json_data:
        raise ValueError("No data to validate.")

    embedding_length = len(json_data[0]['embeddings'])
    for record in json_data:
        if not all(isinstance(x, float) for x in record['embeddings']):
            raise ValueError("Embeddings must be floats.")
        if len(record['embeddings']) != embedding_length:
            raise ValueError("Inconsistent embedding dimensions found.")

def init_pinecone() -> None:
    """
    Initializes the Pinecone environment.

    Raises:
        ValueError: If Pinecone API key is not found in environment variables.
    """
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in the environment variables.")

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

def replace_or_create_pinecone_index(index_name: str, dimension: int, metric: str = 'cosine') -> pinecone.Index:
    """
    Replaces an existing Pinecone index with a new one or creates it if it doesn't exist.

    This function first checks if an index with the given name already exists.
    If it does, the existing index is deleted. Then, a new index with the specified
    parameters is created.

    Args:
        index_name (str): The name of the index to create or replace.
        dimension (int): The dimension of the embeddings to be stored in the index.
        metric (str, optional): The type of metric used in the vector index. Options
                                are "cosine", "dotproduct", "euclidean". Defaults to "cosine".

    Returns:
        pinecone.Index: An object representing the newly created Pinecone index.

    Note:
        This function will delete an existing index with the same name, which might
        lead to loss of data. Ensure that this behavior is intended before using this function.
    """
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
        logging.info(f"Existing Pinecone index '{index_name}' found and deleted.")

    pinecone.create_index(index_name, dimension=dimension, metric=metric)
    logging.info(f"New Pineconce index '{index_name}' created with dimension {dimension} and metric {metric}.")

    return pinecone.Index(index_name)


def process_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes the metadata of a record, ensuring all values are of the correct type.

    Args:
        record (Dict[str, Any]): The record containing the metadata.

    Returns:
        Dict[str, Any]: Processed metadata with valid types.

    Raises:
        ValueError: If metadata types are invalid.
    """
    meta = {}
    for key, value in record.items():
        if key != 'embeddings':
            if value is None:
                continue  # Skip null values
            elif isinstance(value, list):
                meta[key] = [str(v) for v in value]
            elif isinstance(value, (str, int, float, bool)):
                meta[key] = value
            else:
                raise ValueError(f"Invalid metadata type for key '{key}': {type(value)}")
            
    logging.debug("Metadata processed successfully.")
    return meta

def prepare_upsert_data(json_data: List[Dict[str, Any]]) -> List[Tuple[str, List[float], Dict[str, Any]]]:
    """
    Prepares the data for upserting into Pinecone, ensuring metadata values are of the correct type.

    Args:
        json_data (List[Dict[str, Any]]): The JSON data containing the embeddings and metadata.

    Returns:
        List[Tuple[str, List[float], Dict[str, Any]]]: Data formatted for upserting.

    Note:
        Ignores items with invalid metadata types and logs errors for them. Reports the count of processed and ignored items.
    """
    prepared_data = []
    ignored_count = 0

    for i, record in enumerate(json_data):
        try:
            meta = process_metadata(record)
            prepared_data.append((str(i), record['embeddings'], meta))
        except ValueError as e:
            logging.error(f"Error processing record {i}: {e}")
            ignored_count += 1

    processed_count = len(prepared_data)
    logging.info(f"Data preparation completed. Processed: {processed_count}, Ignored: {ignored_count}")
    
    return prepared_data


def batch_upsert(index: pinecone.Index, data: List[Tuple[str, List[float], Dict[str, Any]]], batch_size: int = 128, one_by_one: bool = True) -> None:
    """
    Batch upsert data into a Pinecone index, with an option to handle upserts individually or in batches.

    Args:
        index (pinecone.Index): The Pinecone index.
        data (List[Tuple[str, List[float], Dict[str, Any]]]): The data to upsert.
        batch_size (int): The size of each batch.
        one_by_one (bool): If True, upserts items one by one for detailed error logging.
    """
    success_count = 0
    fail_count = 0

    for i in tqdm(range(0, len(data), batch_size)):
        batch_data = data[i:i + batch_size]

        if one_by_one:
            # Upsert items one by one for detailed error logging
            for item in batch_data:
                try:
                    index.upsert(vectors=[item])
                    success_count += 1
                except Exception as e:
                    logging.error(f"Failed to upsert item {item[0]}: {e}")
                    fail_count += 1
        else:
            # Upsert in batches for efficiency
            try:
                index.upsert(vectors=batch_data)
                success_count += len(batch_data)
            except Exception as e:
                logging.error(f"Failed to upsert batch starting with item {batch_data[0][0]}: {e}")
                fail_count += len(batch_data)

    logging.info(f"Upsert completed. Success: {success_count}, Failed: {fail_count}")


def main() -> None:
    """
    Main function to execute the script logic.
    """
    logging.basicConfig(level=logging.INFO)
    setup_global_logger()
    load_dotenv()

    embeddings_file_path = os.getenv('EMBEDDINGS_DATA_FILE_PATH')
    if not embeddings_file_path:
        raise ValueError("EMBEDDINGS_DATA_FILE_PATH not found in the environment variables.")

    try:
        json_embeddings_data = read_json_file(embeddings_file_path)
        validate_embeddings(json_embeddings_data)
    except Exception as e:
        raise ValueError(f"Error processing file: {e}")

    init_pinecone()

    index_name = os.getenv('PINECONE_INDEX_NAME')
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME not found in the environment variables.")
    embeddings_dimensions = len(json_embeddings_data[0]['embeddings'])
    index = replace_or_create_pinecone_index(index_name, embeddings_dimensions)

    upsert_data = prepare_upsert_data(json_embeddings_data)
    batch_upsert(index, upsert_data, batch_size=1)

    index_stats = index.describe_index_stats()

    namespace_details = ', '.join([f"Namespace '{ns}': {stats['vector_count']} vectors" for ns, stats in index_stats['namespaces'].items()])

    log_message = (
        f"Pinecone Index Statistics:\n"
        f" - Dimensionality: {index_stats['dimension']} (Each vector has {index_stats['dimension']} elements)\n"
        f" - Index Fullness: {index_stats['index_fullness']:.2%} (Percentage of the index's total capacity used)\n"
        f" - Total Vector Count: {index_stats['total_vector_count']} (Total number of vectors stored in the index)\n"
        f" - Namespaces: {len(index_stats['namespaces'])} (Number of distinct namespaces)\n"
        f"   - Details: {namespace_details}"
    )

    logging.info(log_message)


if __name__ == "__main__":
    main()