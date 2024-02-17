# TODO: Ensure the tests are comprehensive

import pytest
from unittest.mock import patch
from services.chatbot_interactor import generate_chat_response


class TestGenerateChatResponse:
    @pytest.mark.parametrize(
        "model_name,expected_result",
        [
            ("model_1", "Expected response from model_1"),
            ("model_2", "Expected response from model_2"),
        ],
    )
    def test_generate_chat_response_valid_inputs(self, mocker, model_name, expected_result):
        # Mock dependencies
        mock_prompt_template = mocker.patch(
            "services.chatbot_interactor.ChatPromptTemplate.from_template",
            return_value="Mocked Prompt",
        )
        mock_model = mocker.patch("services.chatbot_interactor.ChatOpenAI", autospec=True)
        mock_output_parser = mocker.patch(
            "services.chatbot_interactor.StrOutputParser", autospec=True
        )

        # Mock the chain to return the expected result
        # Ensure the mock chain's final invoke method returns the expected result
        mock_model.return_value.__ror__.return_value.__or__.return_value.invoke.return_value = (
            expected_result
        )

        # Test data
        query_results = "Some information"
        user_query = "A question?"
        url = "http://example.com"

        result = generate_chat_response(query_results, user_query, url, model_name)

        assert (
            result == expected_result
        ), "The function should return the expected response based on the model_name."

    def test_generate_chat_response_handles_exceptions(self, mocker):
        # Mock dependencies to raise an exception
        mocker.patch(
            "services.chatbot_interactor.ChatPromptTemplate.from_template",
            side_effect=Exception("Test Exception"),
        )

        # Test data
        query_results = "Some information"
        user_query = "A question?"
        url = "http://example.com"
        model_name = "model_1"

        # Expect the function to handle exceptions gracefully
        with pytest.raises(Exception) as exc_info:
            generate_chat_response(query_results, user_query, url, model_name)

        assert "Test Exception" in str(
            exc_info.value
        ), "The function should raise the expected exception."
