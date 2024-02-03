import logging
import os
import pinecone
import re
from dotenv import load_dotenv
from config.logging_config import setup_global_logger
from utils.utils import read_json_file, read_yaml_file, load_embeddings, join_data, validate_embeddings
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from datetime import datetime


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


def process_metadata(record: Dict[str, Any], metadata_to_extract: List[str] = None) -> Dict[str, Any]:
    """
    Processes the metadata of a record. If 'metadata_to_extract' is provided, 
    extracts only the specified metadata fields. Otherwise, processes all fields in the record.

    Args:
        record (Dict[str, Any]): The record containing the metadata.
        metadata_to_extract (List[str], optional): List of the metadata attributes to extract.
            If None, all metadata fields in the record are processed.

    Returns:
        Dict[str, Any]: Processed metadata with valid types.

    Raises:
        ValueError: If metadata types are invalid.
    """
    meta = {}

    for key, value in record.items():
        if metadata_to_extract and key not in metadata_to_extract:
            continue 

        if key != 'embeddings':
            if value is None:
                continue  # Skip null values
            elif isinstance(value, list):
                meta[key] = [str(v) for v in value]  # Convert all elements in list to string
            elif isinstance(value, (str, int, float, bool)):
                meta[key] = value
            else:
                raise ValueError(f"Invalid metadata type for key '{key}': {type(value)}")

    logging.debug("Metadata processed successfully.")
    return meta



def prepare_upsert_data(embeddings_data: List[Dict[str, Any]], metadata_to_extract: List[str] = None) -> List[Tuple[str, List[float], Dict[str, Any]]]:
    """
    Prepares the data for upserting into Pinecone, ensuring metadata values are of the correct type.

    Args:
        embeddings_data (List[Dict[str, Any]]): The embeddings_data data containing the embeddings and metadata.
        metadata_to_extract (List[str], optional): List of the metadata attributes to extract.
            If None, all metadata fields in the record are processed.

    Returns:
        List[Tuple[str, List[float], Dict[str, Any]]]: Data formatted for upserting.

    Note:
        Ignores items with invalid metadata types and logs errors for them. Reports the count of processed and ignored items.
    """
    prepared_data = []
    skipped_count = 0

    for i, record in enumerate(embeddings_data):
        try:
            metadata = process_metadata(record, metadata_to_extract)
            embedding = tuple(record['embedding'])
            id = str(i)
            prepared_data.append((id, embedding, metadata))
        except ValueError as e:
            logging.error(f"Error processing record {i}: {e}")
            skipped_count += 1

    processed_count = len(prepared_data)
    logging.info(f"Data preparation completed. Processed: {processed_count}, Skipped: {skipped_count}")
    
    return prepared_data


def batch_upsert(index: pinecone.Index, data: List[Tuple[str, List[float], Dict[str, Any]]], batch_size: int = 100, one_by_one: bool = False) -> None:
    """
    Batch upsert data into a Pinecone index, with an option to handle upserts individually or in batches.

    Args:
        index (pinecone.Index): The Pinecone index.
        data (List[Tuple[str, List[float], Dict[str, Any]]]): The data to upsert.
        batch_size (int): The size of each batch. Defaults to 100 as per Pinecone's recommendation.
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


def index_records(
    pinecone_api_key: str,
    pinecone_environment: str,
    pinecone_index_name: str,
    embeddings_data: List[Dict[str, Any]],
    metadata_to_extract: List[str]
) -> None:
    """
    Initializes Pinecone, creates or replaces a Pinecone index, upserts data into the index,
    and logs index statistics.

    Parameters:
    - pinecone_api_key (str): The API key for Pinecone.
    - pinecone_environment (str): The environment for Pinecone.
    - pinecone_index_name (str): The name of the Pinecone index to create or replace.
    - embeddings_data (List[Dict[str, Any]]): A list of dictionaries, where each dictionary contains an 'embedding'
      key with the embedding vector and potentially additional metadata.
    - metadata_to_extract (List[str]): A list of keys specifying which metadata fields to extract from embeddings_data
      and include in the upsert operation.
    TODO: Abstract away the implementation so that Pinecone is loosely coupled with the rest of the code.
    """
    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

    # Creating or replacing a Pinecone index
    embeddings_dimensions = len(embeddings_data[0]['embedding'])
    index = replace_or_create_pinecone_index(pinecone_index_name, embeddings_dimensions)

    # Upserting data into Pinecone
    upsert_data = prepare_upsert_data(embeddings_data, metadata_to_extract)
    batch_upsert(index, upsert_data, batch_size=1, one_by_one=True)

    # Logging Pinecone index statistics
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

def main():
    # TODO: Write the docstring as in respectful_scraper.py
    logging.basicConfig(level=logging.INFO)
    setup_global_logger()
    load_dotenv()

    # Configuration parameters
    config = read_yaml_file('config/parameters.yml')
    embeddings_indexer_config = config['embeddings_indexer']

    embeddings_file_path = embeddings_indexer_config.get('embeddings_file_path')
    embeddings_metadata_file_path = embeddings_indexer_config.get('embeddings_metadata_file_path')
    metadata_to_extract = embeddings_indexer_config.get('metadata_fields_to_extract')
    index_name = embeddings_indexer_config.get('pinecone_index_name')
    pinecone_environment = embeddings_indexer_config.get('pinecone_environment')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')

    logging.info(f"Processing files:\n{embeddings_file_path}\n{embeddings_metadata_file_path}")

    # Loading and validating data
    try:
        embeddings_metadata_json = read_json_file(embeddings_metadata_file_path)
        embeddings_hdf5 = load_embeddings(embeddings_file_path)
        embeddings_data = join_data(embeddings_metadata_json, embeddings_hdf5)
        validate_embeddings(embeddings_data)
    except Exception as e:
        raise ValueError(f"Error processing file: {e}")

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

    # Creating a Pinecone index
    embeddings_dimensions = len(embeddings_data[0]['embedding'])
    index = replace_or_create_pinecone_index(index_name, embeddings_dimensions)

    # Upserting data into Pinecone
    upsert_data = prepare_upsert_data(embeddings_data, metadata_to_extract)
    batch_upsert(index, upsert_data, batch_size=1, one_by_one=True)

    # Logging Pinecone index statistics
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