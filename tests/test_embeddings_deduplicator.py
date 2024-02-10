# TODO: Ensure the tests are comprehensive


import numpy as np
import pytest
import faiss
from services.embeddings_deduplicator import build_faiss_index, validate_embeddings, process_and_sort_duplicates, deduplicate_embeddings

def test_build_faiss_index_l2():
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]])
    index = build_faiss_index(vectors)
    assert index.ntotal == 2  # Check if two vectors are added to the index
    assert isinstance(index, faiss.IndexFlatL2)  # Ensure the index is of type IndexFlatL2 for L2 norm

def test_build_faiss_index_ip():
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]])
    index = build_faiss_index(vectors, use_l2=False)
    assert index.ntotal == 2
    assert isinstance(index, faiss.IndexFlatIP)  # Ensure the index is of type IndexFlatIP for inner product

def test_validate_embeddings_matching():
    original = [{'embedding_id': '1', 'embedding': [0.1, 0.2]}]
    truncated = [{'embedding_id': '1', 'embedding': [0.1, 0.2]}]
    # This should not raise any ValueError
    validate_embeddings(original, truncated)

def test_validate_embeddings_non_matching():
    original = [{'embedding_id': '1', 'embedding': [0.1, 0.2]}]
    truncated = [{'embedding_id': '1', 'embedding': [0.2, 0.3]}]
    with pytest.raises(ValueError):
        validate_embeddings(original, truncated)

def test_process_and_sort_duplicates():
    duplicate_records = [({'url': 'url1', 'detokenized_chunk': 'chunk1', 'extra': 'info'}, {'url': 'url2', 'detokenized_chunk': 'chunk2', 'extra': 'info'}, 0.1)]
    sorted_duplicates = process_and_sort_duplicates(duplicate_records)
    expected = [({'url': 'url1', 'detokenized_chunk': 'chunk1'}, {'url': 'url2', 'detokenized_chunk': 'chunk2'}, 0.1)]
    assert sorted_duplicates == expected, "The process_and_sort_duplicates function does not correctly process the input."

def test_deduplicate_embeddings_no_duplicates():
    records = [{'embedding_id': 1, 'embedding': [0.1, 0.2]}, {'embedding_id': 2, 'embedding': [0.3, 0.4]}]
    unique_records, duplicates = deduplicate_embeddings(records, True, 0.01)
    assert len(unique_records) == 2 and len(duplicates) == 0
    
def test_deduplicate_embeddings_with_duplicates():
    records = [{'embedding_id': 1, 'embedding': [0.1, 0.2]}, {'embedding_id': 2, 'embedding': [0.1, 0.2]}, {'embedding_id': 3, 'embedding': [0.3, 0.4]}]
    unique_records, duplicates = deduplicate_embeddings(records, True, 0.01)
    assert len(unique_records) == 2, "Expected 2 unique records after deduplication"
    assert len(duplicates) == 2, f"Expected 1 duplicate pair (2 duplicate records), got {len(duplicates)}"