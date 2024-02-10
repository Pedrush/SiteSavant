# TODO: Ensure the tests are comprehensive

import pytest
from services.query_handler import parse_query_results, process_query
from unittest.mock import patch

@pytest.mark.parametrize("input_data,expected_output", [
    ({"matches": [{"metadata": {"detokenized_chunk": "Text1"}}]}, "Text1"),
    ({"matches": [{"metadata": {"detokenized_chunk": "Text1"}}, {"metadata": {"detokenized_chunk": "Text2"}}]}, "Text1\n\nText2"),
    ({}, ""),
    ({"matches": []}, ""),
    ({"matches": [{"metadata": {}}]}, ""),
    ({"matches": [{"metadata": {"detokenized_chunk": ""}}]}, ""),
])
def test_parse_query_results(input_data, expected_output):
    assert parse_query_results(input_data) == expected_output


@pytest.fixture
def mock_cohere_service(mocker):
    mock = mocker.patch('services.query_handler.CohereTextProcessingService.get_embedding', return_value=[0.1, 0.2, 0.3])
    return mock

@pytest.fixture
def mock_pinecone_init(mocker):
    return mocker.patch('services.query_handler.pinecone.init')

@pytest.fixture
def mock_pinecone_index(mocker):
    mock = mocker.Mock()
    mocker.patch('services.query_handler.pinecone.Index', return_value=mock)
    mock.query.return_value = {"results": "mocked results"}
    return mock

def test_process_query_with_valid_input(mock_cohere_service, mock_pinecone_init, mock_pinecone_index):
    result = process_query("test query", "model-name", "environment", "index-name", 5, "cohere-api-key", "pinecone-api-key")
    assert result == {"results": "mocked results"}
    mock_cohere_service.assert_called_once()
    mock_pinecone_index.query.assert_called_once()

