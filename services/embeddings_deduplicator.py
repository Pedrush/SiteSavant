"""This script is designed for the deduplication of embeddings, leveraging the power of FAISS
(Facebook AI Similarity Search) for efficient similarity search and numpy for numerical operations.
It aims to identify and remove duplicate records within a dataset based on the similarity of their
embeddings.

Functional Overview:
- Initializes logging and configures settings through environment variables and YAML files.
- Loads embeddings along with their metadata, preparing the dataset for deduplication.
- Utilizes FAISS for building an index of embeddings, enabling efficient similarity searches.
- Identifies duplicate embeddings based on a configurable similarity threshold and metric.
- Processes and removes identified duplicates, ensuring a dataset of unique embeddings.
- Saves the deduplicated dataset and a detailed report of the duplicates for review.

Components:
- `build_faiss_index`: Constructs a FAISS index with the choice of L2 norm or inner product for
  similarity measures.
- `validate_embeddings`: Validates the integrity of embeddings post-deduplication against the
   original dataset.
- `process_and_sort_duplicates`: Sorts identified duplicate records by their similarity measure
   for easier analysis.
- `deduplicate_embeddings`: The core function that orchestrates the deduplication process,
   leveraging FAISS for efficient similarity searches and threshold-based duplicate identification.
- Utility functions for configuration reading, data loading, and result saving are integral to the
  workflow.

Usage:
Can be used as a standalone module. Additionally, the functions are designed to integrate with
a larger processing pipeline, as demonstrated in the main orchestrator script (main.py) within
this project.
"""

# Standard library imports
import logging
from typing import Dict, List, Tuple

# Related third-party imports
import faiss
import numpy as np

# Local application/library specific imports
from config.logging_config import setup_global_logger
from utils.utils import (
    join_data,
    load_embeddings,
    read_json_file,
    read_yaml_file,
    save_embeddings_and_metadata,
    write_json_file,
    generate_timestamp,
)


def build_faiss_index(vectors: np.ndarray, use_l2: bool = True) -> faiss.IndexFlat:
    """Build a FAISS index for the given vectors.

    Args:
        vectors: An array of vectors to index.
        use_l2: Whether to use L2 norm for indexing. Defaults to True.
        If False, inner product is used instead.

    Returns:
        The constructed FAISS index.
    """
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension) if use_l2 else faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index


def validate_embeddings(original: List[Dict[str, any]], truncated: List[Dict[str, any]]) -> None:
    """Verifies that each record in the truncated list has a corresponding record in the original
    list with a matching embedding, using embedding_id as a unique identifier. Raises an error if
    a mismatch or missing embedding_id is found.

    Args:
    - original (List[Dict[str, any]]): The original list of dictionaries containing embedding
      information.
    - truncated (List[Dict[str, any]]): The truncated list of dictionaries to compare against the
      original list.

    Raises:
    - ValueError: If an embedding_id does not have a corresponding record in the original list or
      if the embedding values do not closely match.

    Returns:
    - None
    """
    original_by_id = {item["embedding_id"]: item for item in original}

    for item in truncated:
        embedding_id = item["embedding_id"]

        if embedding_id not in original_by_id:
            raise ValueError(
                f"Unexpected error. embedding_id: {embedding_id} not found in original data."
            )

        if not np.allclose(
            np.array(item["embedding"]),
            np.array(original_by_id[embedding_id]["embedding"]),
            atol=1e-8,
        ):
            raise ValueError(
                f"Embeddings do not closely match for record with embedding_id: {embedding_id}."
            )

    logging.debug("Sanity check passed: All deduplicated embeddings match their original records.")


def process_and_sort_duplicates(
    duplicate_records: List[Tuple[Dict[str, any], Dict[str, any], float]]
) -> List[Tuple[Dict[str, any], Dict[str, any], float]]:
    """Function made to inspect what records were considered duplicate. Extracts only the relevant
    information from the duplicate records and sorts them based on the distance.

    Args:
        duplicate_records (list of tuples): A list containing tuples of (record1, record2, dist).

    Returns:
        list: Sorted list of duplicate records based on the distance.
    """
    for i, (record1, record2, dist) in enumerate(duplicate_records):
        record1 = {k: v for k, v in record1.items() if k in ["detokenized_chunk", "url"]}
        record2 = {k: v for k, v in record2.items() if k in ["detokenized_chunk", "url"]}
        dist = float(dist)
        duplicate_records[i] = (record1, record2, dist)

    sorted_duplicate_records = sorted(duplicate_records, key=lambda x: x[2])
    return sorted_duplicate_records


def deduplicate_embeddings(
    records: List[dict], use_l2_similarity: bool, threshold: float
) -> Tuple[List[dict], List[Tuple[dict, dict, float]]]:
    """Deduplicates records based on the similarity of their embeddings using a FAISS index.

    This function processes a list of records, each containing an embedding, to identify and
    remove duplicates. A FAISS index is constructed from the embeddings, and pairwise similarity
    is computed. Records are considered duplicates if their similarity (L2 distance if
    `use_l2_similarity` is True, inner product otherwise) is below a specified threshold. For each
    pair of duplicate records, the one with the larger index in the list is removed.
    The process repeats until no more duplicates are found. This function also ensures that the
    final set of deduplicated records is consistent with their original metadata.

    Parameters:
    - records (List[dict]): A list of dictionaries, each containing an 'embedding' key with its
      associated vector.
    - use_l2_similarity (bool): If True, use L2 distance as the similarity metric; otherwise,
      inner product is uesed.
    - threshold (float): The distance threshold below which two embeddings are considered
      duplicates.

    Returns:
    - Tuple[List[dict], List[Tuple[dict, dict, float]]]: A tuple where the first element is a list
      of deduplicated records, and the second element is a list of tuples, each representing a
      pair of duplicate records and their similarity score. The second record in each tuple is the
      one that was removed.

    Example usage:
    >>> records = [{'id': 1, 'embedding': [0.1, 0.2]}, {'id': 2, 'embedding': [0.1, 0.2]}, {'id': 3, 'embedding': [0.3, 0.4]}]
    >>> unique_records, duplicates = deduplicate_embeddings(records, True, 0.05)
    """
    original_records = records
    original_records_length = len(records)
    duplicate_records = []
    duplicates_found = True

    while duplicates_found:
        vectors = np.array([record["embedding"] for record in records])
        index = build_faiss_index(vectors, use_l2=use_l2_similarity)
        duplicates_found = False
        distances, indices = index.search(vectors, 2)
        to_remove = set()
        duplicate_pairs = []

        for distance_row, index_row in zip(distances, indices):
            if distance_row[1] < threshold and index_row[1] != -1:
                larger_index = max(index_row[0], index_row[1])
                to_remove.add(larger_index)
                duplicates_found = True
                duplicate_pairs.append((index_row[0], index_row[1], distance_row[1]))

        if duplicates_found:
            vectors = np.delete(vectors, list(to_remove), axis=0)
            duplicate_records.extend(
                [(records[i], records[j], dist) for (i, j, dist) in duplicate_pairs]
            )
            records = [record for i, record in enumerate(records) if i not in to_remove]

    n_of_duplicates = original_records_length - len(records)
    logging.info(
        f"{n_of_duplicates} duplicates found and removed from {original_records_length} records, "
        f"resulting in {len(records)} unique records."
    )
    validate_embeddings(original=original_records, truncated=records)
    return records, duplicate_records


def main():
    """Demonstrates the capabilities of various functions for embeddings deduplication.

    Steps:
    1. Configuration setup, including logging and environment variables.
    2. Reads parameters and file paths from a YAML configuration file.
    3. Loads embeddings and their metadata from specified file paths.
    4. Joins embeddings with metadata to form a comprehensive dataset.
    5. Deduplicates the dataset based on embedding similarity, using configurable parameters for
       similarity metrics and threshold.
    6. Sorts and processes duplicate records for inspection.
    7. Writes the sorted duplicate records to a JSON file with a timestamp for tracking.
    8. Saves the unique records (embeddings and metadata) to a specified directory, also
       timestamped.
    """
    # Config
    logging.basicConfig(level=logging.INFO)
    setup_global_logger()
    timestamp = generate_timestamp()

    all_parameters = read_yaml_file("config/parameters.yml")
    config = all_parameters["main_config"]
    file_paths = all_parameters["file_paths"]

    # Deduplicating embeddings
    embeddings = load_embeddings(
        file_paths["embeddings_deduplicator"]["input_embeddings_file_path"]
    )
    metadata = read_json_file(
        file_paths["embeddings_deduplicator"]["input_embeddings_metadata_file_path"]
    )
    embeddings_with_metadata = join_data(records=metadata, embeddings=embeddings)
    unique_records, duplicate_records = deduplicate_embeddings(
        records=embeddings_with_metadata,
        **config["embeddings_deduplicator"],
    )
    sorted_duplicate_records = process_and_sort_duplicates(duplicate_records)
    write_json_file(
        data=sorted_duplicate_records,
        file_path=file_paths["embeddings_deduplicator"]["output_duplicate_records_file_path"],
        timestamp=timestamp,
    )
    save_embeddings_and_metadata(
        data=unique_records,
        data_dir=file_paths["embeddings_deduplicator"]["output_embeddings_deduplicated_data_dir"],
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
