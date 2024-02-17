"""
This script is architected for managing and indexing embeddings into a Pinecone vector database,
utilizing Pinecone's capabilities for vector similarity search. 3-rd party vector databases, such
as Pinecone, have out-of-the-box support for similarity search, index managenent, efficient data
retrieval, high-dimensional vector storage, and more. This script is designed to levarege these
capabilities.

Functional Overview:

- Initializes logging and configures settings through environment variables and YAML configuration
  files, setting the groundwork for operational transparency and dynamic parameter management.
- Loads embeddings and their associated metadata, preparing the dataset for indexing, ensuring
  that data is ready for insertion into Pinecone's vector database.
- Employs Pinecone for creating or replacing indexes, facilitating efficient vector similarity
  searches with options for various metrics such as cosine similarity, dot product, and Euclidean
  distance.
- Processes and prepares data for upserting into Pinecone, converting metadata to appropriate
  types and structuring data tuples for Pinecone's API.
- Conducts batch upsert operations into the Pinecone index, leveraging Pinecone's batch processing
  capabilities for efficient data insertion and error handling.
- Provides a comprehensive workflow for indexing records, from initialization and configuration to
  data preparation and upserting, demonstrating a full lifecycle of vector database management.

Components:
- replace_or_create_pinecone_index: Manages Pinecone indexes by either creating a new index or
  replacing an existing one. Sufficient for demo purposes.
- process_metadata: Processes and filters metadata associated with embeddings, ensuring
  compatibility and integrity before data insertion into the database.
- prepare_upsert_data: Prepares embeddings and metadata for upsert operations, structuring data in
  a Pinecone-compatible format and ensuring data quality.
- batch_upsert: Handles the insertion of prepared data into Pinecone, offering flexibility in
  batch size and error logging to optimize performance and reliability.
- index_records: Orchestrates the entire process of embedding indexing, from environment setup to
  data upsert, encapsulating the workflow in a single, manageable function.

Usage:
Can be used as a standalone module. Additionally, the functions are designed to integrate with
a larger processing pipeline, as demonstrated in the main orchestrator script (main.py) within
this project.
"""

# Standard library imports
import logging
import os
import time
from typing import Any, Dict, List, Tuple

# Related third-party imports
import pinecone
from dotenv import load_dotenv
from tqdm import tqdm

# Local application/library specific imports
from config.logging_config import setup_global_logger
from utils.utils import join_data, load_embeddings, read_json_file, read_yaml_file


def replace_or_create_pinecone_index(
    index_name: str, dimension: int, metric: str = "cosine"
) -> pinecone.Index:
    """Replaces an existing Pinecone index with a new one or creates it if it doesn't exist.

    This function first checks if an index with the given name already exists.
    If it does, the existing index is deleted. Then, a new index with the specified
    parameters is created. Such approach is compatible with free Pinecone account.

    Args:
        index_name (str): The name of the index to create or replace.
        dimension (int): The dimension of the embeddings to be stored in the index.
        metric (str, optional): The type of metric used in the vector index. Options
                                are "cosine", "dotproduct", "euclidean". Defaults to "cosine".

    Returns:
        pinecone.Index: An object representing the newly created Pinecone index.

    Note:
        This function will delete an existing index with the same name, which might
        lead to loss of data.
    TODO: Implement a more sophisticated index management strategy.
    TODO: Instead of time.sleep, use a more robust method to ensure the index is created.
    """
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
        logging.info(f"Existing Pinecone index '{index_name}' found and deleted.")

    pinecone.create_index(index_name, dimension=dimension, metric=metric)
    logging.info(
        f"New Pineconce index '{index_name}' created with dimension {dimension} and "
        "metric {metric}."
    )
    time.sleep(15)
    return pinecone.Index(index_name)


def process_metadata(
    record: Dict[str, Any], metadata_to_extract: List[str] = None
) -> Dict[str, Any]:
    """Processes the metadata of a record. If 'metadata_to_extract' is provided, extracts only the
    specified metadata fields. Otherwise, processes all fields in the record.

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

        if key != "embeddings":
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


def prepare_upsert_data(
    embeddings_data: List[Dict[str, Any]], metadata_to_extract: List[str] = None
) -> List[Tuple[str, List[float], Dict[str, Any]]]:
    """Prepares the data for upserting into Pinecone, ensuring metadata values are of the correct
    type.

    Args:
        embeddings_data (List[Dict[str, Any]]): The embeddings_data data containing the embeddings
         and metadata.
        metadata_to_extract (List[str], optional): List of the metadata attributes to extract.
        If None, all metadata fields in the record are processed.

    Returns:
        List[Tuple[str, List[float], Dict[str, Any]]]: Data formatted for upserting.

    Note:
        Ignores items with invalid metadata types and logs errors for them. Reports the count of
        processed and ignored items.
    """
    prepared_data = []
    skipped_count = 0

    for i, record in enumerate(embeddings_data):
        try:
            metadata = process_metadata(record, metadata_to_extract)
            embedding = tuple(record["embedding"])
            id = str(i)
            prepared_data.append((id, embedding, metadata))
        except ValueError as e:
            logging.error(f"Error processing record {i}: {e}")
            skipped_count += 1

    processed_count = len(prepared_data)
    logging.info(
        f"Data preparation completed. Processed: {processed_count}, Skipped: {skipped_count}"
    )

    return prepared_data


def batch_upsert(
    index: pinecone.Index,
    data: List[Tuple[str, List[float], Dict[str, Any]]],
    batch_size: int = 100,
    one_by_one: bool = False,
) -> None:
    """Batch upsert data into a Pinecone index, with an option to handle upserts individually or
    in batches.

    Args:
        index (pinecone.Index): The Pinecone index.
        data (List[Tuple[str, List[float], Dict[str, Any]]]): The data to upsert.
        batch_size (int): The size of each batch. Defaults to 100 as per Pinecone's recommendation.
        one_by_one (bool): If True, upserts items one by one for detailed error logging.
    """
    success_count = 0
    fail_count = 0

    for i in tqdm(range(0, len(data), batch_size)):
        batch_data = data[i : i + batch_size]

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
    embeddings_data: List[Dict[str, Any]],
    pinecone_environment: str,
    pinecone_index_name: str,
    metadata_to_extract: List[str],
    pinecone_api_key: str,
) -> None:
    """Initializes Pinecone, creates or replaces a Pinecone index, upserts data into the index,
    and logs index statistics.

    Parameters:
    - embeddings_data (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
      contains an 'embedding' key with the embedding vector and potentially additional metadata.
    - pinecone_environment (str): The environment for Pinecone.
    - pinecone_index_name (str): The name of the Pinecone index to create or replace.
    - metadata_to_extract (List[str]): A list of keys specifying which metadata fields to extract
      from embeddings_data and include in the upsert operation.
    - pinecone_api_key (str): The API key for Pinecone.

    TODO: Abstract away the implementation so that Pinecone is loosely coupled with the rest of
          the code.
    """
    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

    # Creating or replacing a Pinecone index
    embeddings_dimensions = len(embeddings_data[0]["embedding"])
    index = replace_or_create_pinecone_index(pinecone_index_name, embeddings_dimensions)

    # Upserting data into Pinecone
    upsert_data = prepare_upsert_data(embeddings_data, metadata_to_extract)
    # TODO: Make batch_size and one_by_one configurable from the configuration file
    batch_upsert(index, upsert_data, batch_size=100, one_by_one=False)

    # Logging Pinecone index statistics
    index_stats = index.describe_index_stats()
    namespace_details = ", ".join(
        [
            f"Namespace '{ns}': {stats['vector_count']} vectors"
            for ns, stats in index_stats["namespaces"].items()
        ]
    )
    log_message = (
        f"Pinecone Index Statistics:\n"
        f" - Dimensionality: {index_stats['dimension']} (Each vector has "
        f"{index_stats['dimension']} elements)\n"
        f" - Index Fullness: {index_stats['index_fullness']:.2%} (Percentage of the index's total "
        "capacity used)\n"
        f" - Total Vector Count: {index_stats['total_vector_count']} (Total number of vectors "
        "stored in the index)\n"
        f" - Namespaces: {len(index_stats['namespaces'])} (Number of distinct namespaces)\n"
        f"   - Details: {namespace_details}"
    )

    logging.info(log_message)


def main():
    """Demonstrates the capabilities of various functions for indexing embeddings and their
    metadata into a Pinecone vector database.

    The process includes the following steps:
    1. Setup the global logger for logging purposes.
    2. Load environment variables, particularly the Pinecone API key.
    3. Read the application's configuration and file paths from a YAML file.
    4. Load embeddings and their corresponding metadata from specified file paths.
    5. Join the embeddings and metadata into a single data structure.
    6. Initialize the Pinecone environment and index the prepared data.
    """
    logging.basicConfig(level=logging.INFO)
    setup_global_logger()
    load_dotenv()

    all_parameters = read_yaml_file("config/parameters.yml")
    config = all_parameters["main_config"]
    file_paths = all_parameters["file_paths"]

    # Indexing embeddings
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    embeddings = load_embeddings(file_paths["embeddings_indexer"]["input_embeddings_file_path"])
    metadata = read_json_file(
        file_paths["embeddings_indexer"]["input_embeddings_metadata_file_path"]
    )
    embeddings_with_metadata = join_data(records=metadata, embeddings=embeddings)
    index_records(
        embeddings_data=embeddings_with_metadata,
        **config["embeddings_indexer"],
        pinecone_api_key=pinecone_api_key,
    )


if __name__ == "__main__":
    main()
