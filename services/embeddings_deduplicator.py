import numpy as np
import faiss
import logging
import os
import json
import h5py
from dotenv import load_dotenv
from config.logging_config import setup_global_logger
from typing import List, Tuple, Callable, Dict, Any
from utils.utils import read_json_file, read_yaml_file, load_embeddings, join_data, validate_embeddings, save_embeddings_and_metadata, write_json_file


def build_faiss_index(vectors: np.ndarray, use_l2: bool = True) -> faiss.IndexFlat:
    """
    Build a FAISS index for the given vectors.

    Args:
        vectors: An array of vectors to index.
        use_l2: Whether to use L2 norm for indexing. Defaults to True.

    Returns:
        The constructed FAISS index.
    """
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension) if use_l2 else faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index

def deduplicate_embeddings(records: List[dict], use_l2_similarity: bool, threshold: float) -> Tuple[List[dict], List[Tuple[dict, dict, float]]]:
    """
    Identify and remove duplicate records based on embedding similarity using FAISS.

    This function iterates over a list of records, each containing an 'embedding' field.
    It builds a FAISS index based on these embeddings and finds duplicates.
    Two records are considered duplicates if the distance between their embeddings
    (using L2 or other similarity based on `use_l2_similarity`) is less than a specified threshold. 
    In each pair of duplicates, the record with the larger index is removed. 
    The function iterates until no more duplicates are found.

    Parameters:
    - records (List[dict]): A list of dictionaries, where each dictionary represents a record and must contain an 'embedding' key with a vector value.
    - use_l2_similarity (bool): Flag to use L2 similarity for FAISS index. If False, another similarity metric is used.
    - threshold (float): A threshold for determining duplicates. Two embeddings are considered duplicates if their distance is less than this threshold.

    Returns:
    - Tuple[List[dict], List[Tuple[dict, dict, float]]]: A tuple containing two elements:
        1. A list of dictionaries, representing the unique records after removing duplicates.
        2. A list of tuples (record[i], record[j], distance), where each tuple represents a pair
        of indices from the original list identified as duplicates, along with their distance. The record at index j (the larger index) is removed.

    Example:
    >>> records = [{'id': 1, 'embedding': [0.1, 0.2]}, {'id': 2, 'embedding': [0.1, 0.2]}, {'id': 3, 'embedding': [0.3, 0.4]}]
    >>> unique_records, duplicates = embeddings_deduplicator(records, True, 0.05)
    TODO: Optimise computationally
    TODO: update docstring
    TODO: test if use_l2_similarity is working
    """
    original_records_length = len(records)
    duplicate_records = []
    duplicates_found = True

    while duplicates_found:
        vectors = np.array([record['embedding'] for record in records])
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
            duplicate_records.extend([(records[i], records[j], dist) for (i, j, dist) in duplicate_pairs])
            records = [record for i, record in enumerate(records) if i not in to_remove]

    n_of_duplicates = original_records_length - len(records)
    logging.info(f'{n_of_duplicates} duplicates found and removed from {original_records_length} records, resulting in {len(records)} unique records.')
    return records, duplicate_records

# TODO: delete this function
def check_faiss_alignment(records: List[dict], index: faiss.IndexFlat) -> bool:
    """
    Check if the vectors in the FAISS index correspond to the vectors in the records list.

    Args:
    records (List[dict]): List of records, each containing an 'embedding' field.
    index (faiss.IndexFlat): FAISS index containing the vectors.

    Returns:
    bool: True if the alignment is correct, False otherwise.
    """
    
    # Retrieve a few sample vectors from the FAISS index
    sample_indices = np.random.choice(len(records), size=min(10, len(records)), replace=False)
    for i in sample_indices:
        i = int(i)  # Ensure the index is an integer
        # Retrieve the vector from the FAISS index
        faiss_vector = np.zeros(index.d, dtype=np.float32)
        index.reconstruct(i, faiss_vector)

        # Compare with the corresponding vector in the records list
        record_vector = np.array(records[i]['embedding'], dtype=np.float32)
        if not np.allclose(faiss_vector, record_vector, atol=1e-6):
            return False

    return True


def check_join_success(processed_data: List[Dict[str, any]], original_data: List[Dict[str, any]]) -> None:
    """
    Runtime sanity check function.
    Verifies that each record in processed_data has a corresponding record in original_data with a matching embedding.
    Uses numpy.allclose to compare embeddings for floating-point precision tolerance.

    Parameters:
    processed_data (List[Dict[str, any]]): A list of dictionaries, each representing a processed record.
    original_data (List[Dict[str, any]]): A list of dictionaries, each representing an original record.

    Raises:
    ValueError: If any record in processed_data does not have a corresponding record in original_data with a matching embedding.

    Returns:
    None
    """

    for record in processed_data:
        record_embedding = record['embedding']
        record_id = record['embedding_id']

        # Find the corresponding record in the original data.
        original_record = next((rec for rec in original_data if rec['embedding_id'] == record_id), None)

        # Compare embeddings using numpy.allclose
        if original_record is None:
            raise ValueError(f"Unexpected error. Record embedding_id: {record_id} not found in original data.")
        elif not np.allclose(original_record['embedding'], record_embedding):
            raise ValueError(f"Embeddings do not match for record with embedding_id: {record_id}.")

    # TODO: change to logging.debug
    logging.info("Join of deduplicated embeddings back to the records was successful.")

def process_and_sort_duplicates(duplicate_records):
    """
    Function made to inspect what records were considered duplicate.
    Extracts only the relevant information from the duplicate records
    and sorts them based on the distance.

    Args:
        duplicate_records (list of tuples): A list containing tuples of (record1, record2, dist).

    Returns:
        list: Sorted list of duplicate records based on the distance.
    """
    for i, (record1, record2, dist) in enumerate(duplicate_records):
        record1 = {k: v for k, v in record1.items() if k in ['detokenized_chunk', 'url']}
        record2 = {k: v for k, v in record2.items() if k in ['detokenized_chunk', 'url']}
        dist = float(dist)
        duplicate_records[i] = (record1, record2, dist)

    sorted_duplicate_records = sorted(duplicate_records, key=lambda x: x[2])
    return sorted_duplicate_records



def main():
    logging.basicConfig(level=logging.INFO)
    setup_global_logger()
    load_dotenv()

    # Configuration parameters
    config = read_yaml_file('config/parameters.yml')
    embeddings_deduplicator_config = config['embeddings_deduplicator']

    embeddings_file_path = embeddings_deduplicator_config.get('input_embeddings_file_path')
    embeddings_metadata_file_path = embeddings_deduplicator_config.get('input_embeddings_metadata_file_path')
    use_l2_similarity = embeddings_deduplicator_config.get('use_l2_similarity')
    distance_threshold = embeddings_deduplicator_config.get('distance_threshold')
    output_data_dir = embeddings_deduplicator_config.get('output_data_dir')
    output_metadata_file_name = embeddings_deduplicator_config.get('output_metadata_file_name')
    output_embeddings_file_name = embeddings_deduplicator_config.get('output_embeddings_file_name')

    # Loading and validating data
    try:
        embeddings_metadata_json = read_json_file(embeddings_metadata_file_path)
        embeddings_hdf5 = load_embeddings(embeddings_file_path)
        embeddings_data = join_data(embeddings_metadata_json, embeddings_hdf5)
        validate_embeddings(embeddings_data)
    except Exception as e:
        raise ValueError(f"Error processing file: {e}")

    # TODO: remove root/repos from the path
    logging.info(f"Processing files:\n{embeddings_file_path}\n{embeddings_metadata_file_path}")
    logging.info(f"Deduplicating {len(embeddings_data)} records...")

    unique_records, duplicate_records = deduplicate_embeddings(
        records=embeddings_data,
        use_l2_similarity=use_l2_similarity, 
        threshold=distance_threshold
        )

    # Check alignment between original records and processed records
    check_join_success(processed_data=unique_records, original_data=embeddings_data)

    # Check which duplicate_records were considered duplicates
    # TODO: remove or run only in debug mode
    for i, (record1, record2, dist) in enumerate(duplicate_records):
        record1 = {k: v for k, v in record1.items() if k in ['detokenized_chunk', 'url']}
        record2 = {k: v for k, v in record2.items() if k in ['detokenized_chunk', 'url']}
        dist = float(dist)
        duplicate_records[i] = (record1, record2, dist)
    
    sorted_duplicate_records = sorted(duplicate_records, key=lambda x: x[2])
    write_json_file(sorted_duplicate_records, 'data/debug/duplicate_records.json')

    # Save the processed data
    if unique_records:
        save_embeddings_and_metadata(
            data=unique_records,
            data_dir=output_data_dir,
            metadata_file_name=output_metadata_file_name,
            embeddings_file_name=output_embeddings_file_name
            )
    else:
        logging.critical("No data to save. Possibly all records are duplicates.")

if __name__ == '__main__':
    main()
    