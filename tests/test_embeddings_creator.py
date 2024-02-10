# TODO: Ensure the tests are comprehensive

import pytest
import requests
from services.embeddings_creator import CohereTextProcessingService, chunk_tokens, embed_file_contents
from unittest.mock import MagicMock

# Mock the Cohere API responses for tokenize, detokenize, and embed endpoints
TOKENIZE_RESPONSE = {"tokens": [101, 102, 103]}
DETOKENIZE_RESPONSE = {"text": "example text"}
EMBEDDING_RESPONSE = {"embeddings": [[0.1, 0.2, 0.3]]}

@pytest.fixture
def cohere_service(requests_mock):
    requests_mock.post("https://api.cohere.ai/v1/tokenize", json={"tokens": [101, 102, 103]})
    requests_mock.post("https://api.cohere.ai/v1/detokenize", json={"text": "example text"})
    requests_mock.post("https://api.cohere.ai/v1/embed", json={"embeddings": [[0.1, 0.2, 0.3]]})
    session = requests.Session()
    return CohereTextProcessingService(session)

def test_tokenize_text_success(cohere_service):
    tokens = cohere_service.tokenize_text("test text")
    assert tokens == TOKENIZE_RESPONSE["tokens"], "The token list should match the mocked response"

def test_detokenize_text_success(cohere_service):
    text = cohere_service.detokenize_text([101, 102, 103])
    assert text == DETOKENIZE_RESPONSE["text"], "The detokenized text should match the mocked response"

def test_get_embedding_success(cohere_service):
    embedding = cohere_service.get_embedding("test text")
    assert embedding == EMBEDDING_RESPONSE["embeddings"][0], "The embedding should match the mocked response"

def test_chunk_tokens():
    tokens = list(range(20))
    max_size = 10
    min_size = 5
    chunks = chunk_tokens(tokens, max_size, min_size)
    assert len(chunks) == 2, "Should split into two chunks"
    assert all(len(chunk) <= max_size for chunk in chunks), "Each chunk size should not exceed max_size"

# TODO: Fix the test
def test_embed_file_contents_success(cohere_service):
    json_data = [{"text": "test text"}]
    processed_data = embed_file_contents(json_data, cohere_service)
    assert len(processed_data) == 1, "Should process one record"
    assert "embedding" in processed_data[0], "Processed data should contain embeddings"

# Example of a test that handles expected exceptions or errors
def test_tokenize_text_exceeds_limit(cohere_service, caplog):
    long_text = "a" * 65537  # Exceeds the mock limit set in the tokenize_text method
    cohere_service.tokenize_text(long_text)
    assert "Text length exceeds the maximum limit" in caplog.text, "Should log a warning for exceeding max length"

# Add more tests as necessary to cover other methods, edge cases, and error handling.
