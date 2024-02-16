"""
This script facilitates the generation of chatbot responses by integrating user queries with model-generated responses based on information retrieved from specified URLs.

Functional Overview:
- Sets up logging and environment configurations for application execution and parameter management.
- Reads application configurations and content from YAML and markdown files.
- Constructs a prompt template to inform the chat model of its role as a digital assistant.
- Utilizes a chat model to process the prompt and generate a response.
- Orchestrates the process flow from configuration and content loading to chat response generation.

Components:
- generate_chat_response: Crafts a chat prompt based on provided information and a user query, processes it through a specified chat model, and returns the model-generated response.

Usage:
Can be used as a standalone module. Additionally, the functions are designed to integrate with a larger processing pipeline, 
as demonstrated in the main orchestrator script (main.py) within this project.
"""

# Standard library imports
import logging

# Related third-party imports
from dotenv import load_dotenv

# Local application/library specific imports
from config.logging_config import setup_global_logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from utils.utils import read_markdown_file, read_yaml_file


def generate_chat_response(query_results: str, user_query: str, url: str, model_name: str) -> str:
    """
    Generates a chat response based on provided information and a user query using a specified model.

    Constructs a prompt template indicating that the assistant is based on information retrieved from
    a specific URL, processes the prompt through a chat model, and returns the generated response.

    Parameters:
    - query_results (str): The information retrieved from the specified URL, formatted as a string.
    - user_query (str): The user's query that needs to be answered based on the provided information.
    - url (str): The URL from which the information was sourced.
    - model_name (str): The name of the model to use for generating the chat response.

    Returns:
    - str: The generated response to the user's query.
    TODO: url can be str or list of str
    TODO: consider removing the url argument, since it's not used in the prompt
    TODO: make the prompt as a system messsage not user message
    """

    prompt = ChatPromptTemplate.from_template(
        "You're SiteSavant, a digital assistant trained to give answers based on specific "
        "information retrieved from websites. If the provided information is insufficient "
        "to answer the query, tell the user that you don't know the answer and suggest what to do next. "

        "Provided Information:\n"
        "{query_results}\n"
        "Based on relevant information, answer completely and accurately only the following query:\n{user_query}"
    )

    model = ChatOpenAI(model=model_name)
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    result = chain.invoke({"url": url, "query_results": query_results, "user_query": user_query})

    return result

def main():
    """
    Demonstrates the capabilities for generating a chatbot response using a combination
    of a  query result and a user query.

    Steps performed by this function include:
    1. Setting up the global logger for capturing the application's operational logs.
    2. Loading environment variables which may include API keys or other configuration not stored in the YAML file.
    3. Reading the application's configuration from a 'parameters.yml' file to get the model name and other settings for the chat response generation.
    4. Reading a markdown file that contains the query results to be used as the basis for generating the chat response.
    5. Invoking the 'generate_chat_response' function with the query result, user query, and other parameters to generate a response.
    6. Logging the generated chatbot response.
    """

    logging.basicConfig(level=logging.INFO)
    setup_global_logger()
    load_dotenv()

    all_parameters = read_yaml_file('config/parameters.yml')
    config = all_parameters['main_config']
    file_paths = all_parameters['file_paths']

    # Generate chatbot interaction
    query_result = read_markdown_file(file_paths['chatbot_interactor']['input_query_results_file_path'])
    chat_response = generate_chat_response(
        query_results=query_result,
        **config['chatbot_interactor']
    )
    logging.info(f"Chatbot response: {chat_response}")

if __name__ == "__main__":
    main()